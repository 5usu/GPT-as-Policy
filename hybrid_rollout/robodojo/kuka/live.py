"""The Qwen -> pi0.5 -> Astra architecture, running against the REAL arm.

WHAT THIS IS
  A real RSI session on the real controller, with real measured joint angles
  and real camera frames driving the full three-model loop. Everything is live
  except the last step: the command is computed, validated, logged -- and not
  sent. The arm holds position for the entire run.

WHY IT EXISTS RATHER THAN A MOTION MODE
  Motion needs numbers nobody has measured yet. The Ruckig limits for this cell
  are absent, so otg.py raises LimitsMissing rather than substitute a plausible
  jerk limit, and rsi_gateway still defaults to linear interpolation, which is
  velocity-discontinuous across a 30 Hz waypoint boundary. Running this mode is
  how those numbers get measured, so it is the step BEFORE motion, not a
  substitute for it.

  It also answers the question that actually decides whether this architecture
  is viable at all, and that no offline run can answer: RSI needs a reply every
  4 ms, Qwen takes seconds and Astra takes minutes. This measures what the RSI
  loop does while the models think.

WHY MOTION CANNOT HAPPEN HERE
  Three independent reasons, not one:
    1. the controller is built with allow_motion=False;
    2. submit() is never called, so no trajectory is ever queued -- the reply
       path has nothing to send but the measured pose;
    3. every reply carries STOPFLAG=1.
  Removing any one of them still leaves the other two.

THREADING
  The RSI loop owns the main thread and must never block: a late reply faults
  the controller. The model loop runs in a worker and only ever READS the
  latest measured state. It cannot write a command, because there is no path
  from it to the reply. Model work is HTTP to other processes (pi0.5, Qwen,
  Astra), so the worker sits in IO and releases the GIL rather than starving
  the RSI thread -- that assumption is measured here too, not assumed: see
  rsi_rate_while_thinking in the summary.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .contract import ARM_DIM

SCHEMA = "kuka.live.v1"

#: Reported when the RSI rate falls below this while the models are running.
#: 250 Hz nominal; sustained loss of a fifth of the cycles is a real finding.
RSI_HEALTHY_HZ = 200.0


@dataclass
class SharedState:
    """Measured state, written by the RSI thread and read by the model thread.

    Deliberately tiny. The model side gets a SNAPSHOT and never a reference, so
    it cannot observe a half-updated pose, and it has no route back.
    """
    lock: threading.Lock = field(default_factory=threading.Lock)
    joints: list[float] | None = None
    ipoc: int | None = None
    updated_at: float = 0.0
    frames_in: int = 0

    def publish(self, joints: Sequence[float], ipoc: int, now: float) -> None:
        with self.lock:
            self.joints = [float(v) for v in joints]
            self.ipoc = int(ipoc)
            self.updated_at = now
            self.frames_in += 1

    def snapshot(self) -> tuple[list[float] | None, int | None, float]:
        with self.lock:
            return (list(self.joints) if self.joints else None,
                    self.ipoc, self.updated_at)


class MotionAttempted(RuntimeError):
    """Raised if anything tries to command motion from this mode."""


def assert_cannot_move(controller: Any) -> None:
    """Fail loudly at construction rather than quietly at 250 Hz."""
    if getattr(controller, "allow_motion", False):
        raise MotionAttempted(
            "live observation mode was handed a controller with "
            "allow_motion=True. This mode exists precisely because the cell "
            "has no measured Ruckig limits and no deviation monitor; it must "
            "not be the thing that moves the arm.")


@dataclass
class ModelCycle:
    """One pass of the three-model loop. Append-only audit row."""
    cycle: int
    started_at: float
    state_age_s: float
    pi05_s: float | None = None
    pipeline_s: float | None = None
    total_s: float | None = None
    proposed_chunk_rows: int | None = None
    controller: str | None = None
    event: str | None = None
    escalated: bool = False
    astra_called: bool = False
    error: str | None = None
    would_have_commanded: list[float] | None = None

    def to_log(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["schema"] = SCHEMA
        d["sent_to_robot"] = False      # invariant of this mode, stated per row
        return d


class LiveObservationRun:
    """Joins the real RSI session to the three-model loop, read-only.

    `pi05_infer(observation) -> 50x7 rows` and `pipeline_step(...)` are injected
    so this module needs no opinion about how pi0.5 is served or how the monitor
    is reached.
    """

    def __init__(self, controller: Any, *,
                 pi05_infer: Callable[[dict[str, Any]], Any],
                 pipeline: Any,
                 grab_frames: Callable[[], tuple[list[str], dict[str, Any]]] | None = None,
                 audit_path: str | None = None,
                 min_model_interval_s: float = 1.0,
                 task: str = "",
                 on_cycle: Callable[[ModelCycle], None] | None = None) -> None:
        assert_cannot_move(controller)
        self.ctrl = controller
        self.pi05_infer = pi05_infer
        self.pipeline = pipeline
        self.grab_frames = grab_frames
        self.audit_path = audit_path
        self.min_model_interval_s = max(0.0, float(min_model_interval_s))
        self.task = task
        self.on_cycle = on_cycle

        self.shared = SharedState()
        self.cycles: list[ModelCycle] = []
        self._stop = threading.Event()
        self._thinking = threading.Event()
        # frames answered while the model loop was mid-inference, and how long
        # that window lasted -- the numbers that decide architectural viability
        self.frames_while_thinking = 0
        self.thinking_seconds = 0.0
        self.first_ipoc: int | None = None
        self.rsi_errors = 0

    # ------------------------------------------------------------ RSI thread
    def serve_one(self, timeout_s: float = 0.05) -> Any:
        """One RSI exchange. Holds position. Never sends a command."""
        out = self.ctrl.serve_cycle(timeout_s=timeout_s)
        if out is None:
            return None
        now = time.time()
        if out.measured:
            self.shared.publish(out.measured, out.ipoc, now)
        if self.first_ipoc is None:
            self.first_ipoc = out.ipoc
        if self._thinking.is_set():
            self.frames_while_thinking += 1
        return out

    # ---------------------------------------------------------- model thread
    def _model_loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            joints, ipoc, updated = self.shared.snapshot()
            if joints is None:
                self._stop.wait(0.1)
                continue
            n += 1
            started = time.time()
            rec = ModelCycle(cycle=n, started_at=started,
                             state_age_s=round(started - updated, 4))
            self._thinking.set()
            think_t0 = time.time()
            try:
                observation = {
                    "observation_id": f"live:{ipoc}",
                    "joints_deg": list(joints),
                    "ipoc": ipoc,
                    "task": self.task,
                }
                urls: list[str] = []
                meta: dict[str, Any] = {}
                if self.grab_frames is not None:
                    urls, meta = self.grab_frames()
                    observation["image_data_urls"] = urls
                    observation["frames_meta"] = meta

                t0 = time.time()
                proposal = self.pi05_infer(observation)
                rec.pi05_s = round(time.time() - t0, 3)
                rows = _rows_of(proposal)
                if rows is None:
                    rec.error = f"pi0.5 returned no usable chunk: {proposal!r}"[:300]
                else:
                    rec.proposed_chunk_rows = len(rows)
                    t1 = time.time()
                    outcome = self.pipeline.step(
                        state=list(joints) + ([0.0] if len(joints) < 7 else []),
                        proposed_steps=len(rows),
                        frames=urls or None,
                        image_data_urls=urls or None,
                        frames_meta=meta or None,
                        proposed_chunk=rows,
                        task=self.task,
                        state_text=_state_text(joints),
                        now=time.time())
                    rec.pipeline_s = round(time.time() - t1, 3)
                    rec.controller = getattr(outcome, "controller", None)
                    rec.event = getattr(outcome, "event", None)
                    rec.escalated = bool(getattr(outcome, "escalated", False))
                    rec.astra_called = bool(getattr(outcome, "astra_called", False))
                    # what WOULD have gone out, had this mode been allowed to
                    # command. Recorded so the gap is inspectable, never sent.
                    rec.would_have_commanded = [round(float(v), 4)
                                                for v in rows[0][:ARM_DIM]]
            except Exception as exc:                            # noqa: BLE001
                rec.error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                self._thinking.clear()
                self.thinking_seconds += time.time() - think_t0
                rec.total_s = round(time.time() - started, 3)
                self.cycles.append(rec)
                self._write(rec)
                if self.on_cycle:
                    try:
                        self.on_cycle(rec)
                    except Exception:                           # noqa: BLE001
                        pass
            self._stop.wait(self.min_model_interval_s)

    def _write(self, rec: ModelCycle) -> None:
        if not self.audit_path:
            return
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_log(), sort_keys=True) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------- lifecycle
    def start_models(self) -> threading.Thread:
        t = threading.Thread(target=self._model_loop, name="models", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()

    def summary(self, elapsed_s: float) -> dict[str, Any]:
        a = self.ctrl.adapter
        done = [c for c in self.cycles if c.error is None]
        lat = [c.total_s for c in done if c.total_s is not None]
        rate_thinking = (self.frames_while_thinking / self.thinking_seconds
                         if self.thinking_seconds > 0 else None)
        return {
            "schema": SCHEMA,
            "motion": "IMPOSSIBLE (allow_motion=False, nothing ever queued, "
                      "every reply STOPFLAG=1)",
            "commands_sent_to_robot": 0,
            "rsi": {"frames_in": a.frames_in, "frames_out": a.frames_out,
                    "malformed": a.malformed, "stale": a.ipoc_regressions,
                    "jumps": a.ipoc_jumps, "last_ipoc": a.last_ipoc,
                    "observed_hz": round(a.frames_in / elapsed_s, 1)
                                   if elapsed_s else None},
            "models": {
                "cycles": len(self.cycles),
                "failed": len(self.cycles) - len(done),
                "mean_cycle_s": round(sum(lat) / len(lat), 3) if lat else None,
                "max_cycle_s": round(max(lat), 3) if lat else None,
                "escalations": sum(1 for c in self.cycles if c.escalated),
                "astra_calls": sum(1 for c in self.cycles if c.astra_called),
            },
            # THE number this mode exists to produce.
            "rsi_rate_while_thinking": (round(rate_thinking, 1)
                                        if rate_thinking else None),
            "rsi_healthy_while_thinking": (
                bool(rate_thinking and rate_thinking >= RSI_HEALTHY_HZ)
                if rate_thinking else None),
            "verdict": _verdict(a, rate_thinking, elapsed_s),
        }


def _verdict(adapter: Any, rate_thinking: float | None,
             elapsed_s: float) -> str:
    if not adapter.frames_in:
        return ("NO RSI FRAMES. The controller never connected; nothing about "
                "the architecture was tested.")
    if adapter.malformed:
        return f"{adapter.malformed} malformed frames -- check the RSI config."
    if rate_thinking is None:
        return ("RSI held, but no model cycle completed, so the timing "
                "question is still unanswered.")
    if rate_thinking < RSI_HEALTHY_HZ:
        return (f"RSI fell to {rate_thinking:.0f} Hz while the models ran. The "
                f"model loop is starving the control loop; this architecture "
                f"is NOT yet safe to give a motion path.")
    return (f"RSI held {rate_thinking:.0f} Hz while the models ran. The control "
            f"loop survives model latency. This is necessary, not sufficient: "
            f"measured Ruckig limits and a deviation monitor are still absent.")


def _rows_of(proposal: Any) -> list[list[float]] | None:
    """Accept either a raw chunk or a {'ok':..,'chunk':..} envelope."""
    if proposal is None:
        return None
    if isinstance(proposal, dict):
        if proposal.get("ok") is False:
            return None
        for key in ("chunk", "actions", "rows", "trajectory"):
            if isinstance(proposal.get(key), list):
                proposal = proposal[key]
                break
        else:
            return None
    if not isinstance(proposal, list) or not proposal:
        return None
    try:
        return [[float(v) for v in row] for row in proposal]
    except (TypeError, ValueError):
        return None


def _state_text(joints: Sequence[float]) -> str:
    return "measured joints (deg): " + ", ".join(
        f"A{i+1}={v:.2f}" for i, v in enumerate(list(joints)[:ARM_DIM]))
