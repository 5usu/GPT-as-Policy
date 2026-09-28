"""The Qwen -> pi0.5 -> Astra architecture on the REAL YAM arms, holding.

The YAM counterpart of `kuka.live`. Everything is live except the last step:
both arms are connected and holding, their joint angles are read continuously,
the three cameras stream, pi0.5 proposes a chunk from the real observation,
the local Qwen monitor triages it and Astra is reached on escalation. The
command is computed and logged -- and not sent. The arms hold throughout.

WHY MOTION CANNOT HAPPEN HERE
  1. the arm reader has no command method (robot.HeldArms);
  2. nothing in this module holds a robot object, only the reader;
  3. `assert_cannot_move` refuses a reader that reports can_move_robot.

WHAT THIS MEASURES
On the KUKA the question was whether RSI keeps its 250 Hz reply while the
models think. i2rt owns its own CAN loop, so here the question is whether the
state stream stays fresh and the arms keep holding while three models run --
`state_rate_while_thinking` -- and how old the state is by the time a chunk
comes back (`state_age_s` per cycle), which bounds how stale every proposal is.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any, Callable

from ..kuka.live import MotionAttempted, ModelCycle, SharedState, _Blind, _rows_of
from .cameras import ENCODE_MIME, frames_to_packet_refs
from .contract import CAMERA_NAMES
from .pipeline import describe_trajectory, state_text

SCHEMA = "yam.live.v1"
#: Below this read rate while the models run, the state stream is not keeping up.
STATE_HEALTHY_HZ = 50.0

__all__ = ["MotionAttempted", "STATE_HEALTHY_HZ", "YamLiveObservationRun",
           "assert_cannot_move", "temporal_frames"]


def assert_cannot_move(arms: Any) -> None:
    if getattr(arms, "can_move_robot", True) or any(
            hasattr(arms, m) for m in ("command_joint_pos", "command", "send")):
        raise MotionAttempted(
            "live observation was handed an arm object that can command motion. "
            "This mode exists to measure the architecture BEFORE a motion path "
            "is given to it.")


def _url(jpeg: bytes) -> str:
    return f"data:{ENCODE_MIME};base64," + base64.b64encode(jpeg).decode()


def temporal_frames(frames: dict, prev: dict) -> tuple[list, dict, dict]:
    """(labelled pairs for the monitor, packet refs, {camera: data URL}).

    Ordered [top t-1, top t, left t-1, left t, ...] so that truncating to the
    monitor's max_frames keeps ONE camera's time pair -- the KUKA finding that
    two viewpoints of one instant cannot answer "is it progressing".
    """
    labelled: list[tuple[str, str]] = []
    for name in CAMERA_NAMES:
        fr = frames.get(name)
        if fr is None or not fr.png:
            continue
        was = prev.get(name)
        if was is not None and was.png:
            labelled.append((f"t-1 {name}", _url(was.png)))
        labelled.append((f"t {name}", _url(fr.png)))
    named = {n: _url(f.png) for n, f in frames.items() if f.png}
    return labelled, frames_to_packet_refs(frames), named


class YamLiveObservationRun:
    """Joins the holding arms and the cameras to the three-model loop, read-only."""

    def __init__(self, arms: Any, *,
                 pi05_infer: Callable[[dict[str, Any]], Any],
                 pipeline: Any,
                 grab_frames: Callable[[], dict] | None = None,
                 fk: Any = None,
                 audit_path: str | None = None,
                 min_model_interval_s: float = 1.0,
                 task: str = "",
                 on_cycle: Callable[[ModelCycle], None] | None = None) -> None:
        assert_cannot_move(arms)
        self.arms = arms
        self.pi05_infer = pi05_infer
        self.pipeline = pipeline
        self.grab_frames = grab_frames
        self.fk = fk
        self.audit_path = audit_path
        self.min_model_interval_s = max(0.0, float(min_model_interval_s))
        self.task = task
        self.on_cycle = on_cycle
        self.shared = SharedState()
        self.cycles: list[ModelCycle] = []
        self._stop = threading.Event()
        self._thinking = threading.Event()
        self._prev_frames: dict = {}
        self.reads = 0
        self.read_errors = 0
        self.reads_while_thinking = 0
        self.thinking_seconds = 0.0
        self.last_measured: list[float] | None = None

    # ------------------------------------------------------------ state loop
    def read_one(self) -> Any:
        """One state read. Publishes the POLICY state (commanded gripper)."""
        try:
            st = self.arms.read()
        except Exception:                                      # noqa: BLE001
            self.read_errors += 1
            return None
        self.reads += 1
        self.last_measured = list(st.measured)
        self.shared.publish(st.policy_state, st.n, time.time())
        if self._thinking.is_set():
            self.reads_while_thinking += 1
        return st

    # ------------------------------------------------------------ model loop
    def run_cycle(self, n: int) -> ModelCycle:
        state, seq, updated = self.shared.snapshot()
        started = time.time()
        rec = ModelCycle(cycle=n, started_at=started,
                         state_age_s=round(started - updated, 4))
        self._thinking.set()
        try:
            if state is None:
                rec.error = "no state read yet"
                raise _Blind(rec.error)
            labelled, meta, named = [], {}, {}
            if self.grab_frames is not None:
                frames = self.grab_frames()
                labelled, meta, named = temporal_frames(frames, self._prev_frames)
                self._prev_frames = dict(frames)
            if set(named) != set(CAMERA_NAMES):
                rec.error = (f"cameras {sorted(set(CAMERA_NAMES) - set(named))} "
                             f"missing; pi0.5 would infer from black frames and "
                             f"the monitor would judge a scene it cannot see")
                raise _Blind(rec.error)
            obs = {"observation_id": f"live:{seq}", "state": list(state),
                   "images": named, "task": self.task}
            t0 = time.time()
            proposal = self.pi05_infer(obs)
            rec.pi05_s = round(time.time() - t0, 3)
            rows = _rows_of(proposal)
            if rows is None:
                rec.error = (f"pi0.5 returned no usable chunk: "
                             f"{(proposal or {}).get('error', proposal)!r}")[:300]
                return rec
            rec.proposed_chunk_rows = len(rows)
            fk_log = self.fk.preview(rows).to_log() if self.fk is not None else None
            t1 = time.time()
            outcome = self.pipeline.step(
                state=list(state), proposed_steps=len(rows),
                frames=labelled or None,
                image_data_urls=[u for _, u in labelled] or None,
                frames_meta=meta or None, proposed_chunk=rows, task=self.task,
                episode_id="live",
                state_text=state_text(state),
                intent_text=describe_trajectory(rows, state),
                fk_preview=fk_log, now=time.time())
            rec.pipeline_s = round(time.time() - t1, 3)
            rec.controller = getattr(outcome, "controller", None)
            rec.event = getattr(outcome, "event", None)
            rec.escalated = bool(getattr(outcome, "escalated", False))
            rec.astra_called = bool(getattr(outcome, "astra_called", False))
            rec.would_have_commanded = [round(float(v), 5) for v in rows[0]]
        except _Blind:
            pass
        except Exception as exc:                               # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self._thinking.clear()
            self.thinking_seconds += time.time() - started
            rec.total_s = round(time.time() - started, 3)
            self.cycles.append(rec)
            self._write(rec)
            if self.on_cycle:
                try:
                    self.on_cycle(rec)
                except Exception:                              # noqa: BLE001
                    pass
        return rec

    def _model_loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            if self.shared.snapshot()[0] is None:
                self._stop.wait(0.05)
                continue
            n += 1
            self.run_cycle(n)
            self._stop.wait(self.min_model_interval_s)

    def _write(self, rec: ModelCycle) -> None:
        if not self.audit_path:
            return
        row = rec.to_log()
        row["schema"] = SCHEMA
        row["robot"] = "yam_bimanual"
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
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
        done = [c for c in self.cycles if c.error is None]
        lat = [c.total_s for c in done if c.total_s is not None]
        ages = [c.state_age_s for c in done]
        rate = (self.reads_while_thinking / self.thinking_seconds
                if self.thinking_seconds > 0 else None)
        return {
            "schema": SCHEMA,
            "motion": "IMPOSSIBLE (hold-only reader, no command method)",
            "commands_sent_to_robot": 0,
            "state": {"reads": self.reads, "read_errors": self.read_errors,
                      "observed_hz": round(self.reads / elapsed_s, 1) if elapsed_s else None},
            "models": {
                "cycles": len(self.cycles), "failed": len(self.cycles) - len(done),
                "errors": sorted({c.error for c in self.cycles if c.error})[:5],
                "mean_cycle_s": round(sum(lat) / len(lat), 3) if lat else None,
                "max_cycle_s": round(max(lat), 3) if lat else None,
                "max_state_age_s": round(max(ages), 3) if ages else None,
                "escalations": sum(1 for c in self.cycles if c.escalated),
                "astra_calls": sum(1 for c in self.cycles if c.astra_called)},
            "state_rate_while_thinking": round(rate, 1) if rate else None,
            "state_healthy_while_thinking": (bool(rate >= STATE_HEALTHY_HZ)
                                             if rate else None),
            "verdict": _verdict(self, rate),
        }


def _verdict(run: YamLiveObservationRun, rate: float | None) -> str:
    if not run.reads:
        return "NO STATE READ. The arms never reported; nothing was tested."
    if rate is None:
        return ("arms held and reported, but no model cycle completed, so the "
                "timing question is still unanswered.")
    if rate < STATE_HEALTHY_HZ:
        return (f"state reads fell to {rate:.0f} Hz while the models ran; the "
                f"model loop is starving the state loop.")
    return (f"state held {rate:.0f} Hz while the models ran. Necessary, not "
            f"sufficient: a measured motion profile and a deviation monitor are "
            f"still absent.")
