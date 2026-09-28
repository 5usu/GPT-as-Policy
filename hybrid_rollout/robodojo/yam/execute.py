"""pi0.5 driving the real YAM arms, with the Qwen gate and Astra escalation.

THIS IS WHERE THE YAM BRANCH GOES FURTHER THAN THE KUKA ONE. The KUKA branch
never gave the policy a motion path (one supervised step at most, and that
never wired). Here, once armed by an operator at the terminal, bounded prefixes
of pi0.5 chunks are executed continuously -- the way the published YAM pi0.5
controller runs this same checkpoint -- through the same deterministic gates:

    observe (both arms + 3 cameras)
      -> pi0.5 chunk (16x14)                       contract-checked per response
      -> validate against the policy state        FATAL -> hold
      -> prefix length: fixed (shadow) or from the Qwen gate / Astra (gating)
      -> sanitize -> re-validate -> displacement cap
      -> motion.MotionOwner: jerk-limited 100 Hz reference, deviation fault

WHY THE MONITOR IS ASYNCHRONOUS HERE
On the Orin a Qwen reading takes 5-9 s (measured on the KUKA branch) and an
Astra review minutes. Either inside the control cycle would freeze the arms for
that long every time the monitor is due. So the pipeline runs in its own thread
on the latest snapshot, and the control cycle reads its latest decision:

  shadow (default)   the monitor and Astra record; the prefix is fixed
  gating             the prefix is the monitor's (or Astra's) clamped number,
                     and a decision older than `monitor_max_age_s` counts as
                     NO decision -> 0 steps -> hold. Silence is never consent.

An Astra `stop` ends the run. A motion fault ends the run. Holding is always
available: 0 steps means the reference stays where it is.

WHAT MUST EXIST BEFORE `execute` WILL ARM
The values in EXECUTE_REQUIRED, measured on this rig, plus an operator typing
the rig id at the terminal. See SETUP.md for how each one is obtained.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .contract import (ACTION_DIM, CHUNK_STEPS, CONTROL_HZ, GRIPPER_INDICES,
                       JOINT_INDICES)
from .live import temporal_frames
from .motion import MotionLimits, MotionRefused, goal_problem
from .pipeline import describe_trajectory, state_text
from .sanitize import sanitize
from .validation import execution_eligible, has_fatal, summarize, validate_chunk

SCHEMA = "yam.execute.v1"

#: What `execute` needs measured on the rig. A subset of safety.REQUIRED_CONFIG:
#: the rest are for the single-step envelope path and Cartesian checks that
#: this joint-space executor does not use.
EXECUTE_REQUIRED = ("camera_mapping", "gripper_limits_left", "gripper_limits_right",
                    "rest_pose", "estop_tested", "max_speed", "max_acceleration",
                    "max_step_displacement", "commanded_observed_tolerance_rad",
                    "observation_freshness_s")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pose(v: Any) -> list[float] | None:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
    if not isinstance(v, list) or len(v) != ACTION_DIM:
        return None
    out = [_num(x) for x in v]
    return None if any(x is None for x in out) else out


def execution_blockers(flat: dict[str, Any]) -> list[str]:
    """Everything that stops `execute` from arming, named. Empty = may arm."""
    out = [f"{k} is not set" for k in EXECUTE_REQUIRED
           if flat.get(k) in (None, "", [], {})]
    if out:
        return out
    speed, acc = _num(flat["max_speed"]), _num(flat["max_acceleration"])
    if speed is None or not 0 < speed <= 2.2:
        out.append("max_speed must be in (0, 2.2] rad/s")
    if acc is None or not 0 < acc <= 6.0:
        out.append("max_acceleration must be in (0, 6] rad/s^2")
    tol = _num(flat["commanded_observed_tolerance_rad"])
    if tol is None or not 0 < tol < 1.0:
        out.append("commanded_observed_tolerance_rad must be in (0, 1) rad")
    disp = _num(flat["max_step_displacement"])
    if disp is None or not 0 < disp <= 1.0:
        out.append("max_step_displacement must be in (0, 1] rad")
    fresh = _num(flat["observation_freshness_s"])
    if fresh is None or not 0 < fresh <= 2.0:
        out.append("observation_freshness_s must be in (0, 2] s")
    for name in ("rest_pose", "start_pose"):
        if name == "start_pose" and flat.get(name) in (None, "", [], {}):
            continue
        p = _pose(flat.get(name))
        if p is None:
            out.append(f"{name} must be {ACTION_DIM} numbers")
        elif goal_problem(p):
            out.append(f"{name}: {goal_problem(p)}")
    return out


def limits_from_config(flat: dict[str, Any]) -> MotionLimits:
    speed, acc = float(flat["max_speed"]), float(flat["max_acceleration"])
    return MotionLimits(velocity=speed, acceleration=acc, jerk=10.0 * acc)


#: For moving to the start and rest poses: slow, whatever the task limits say.
SLOW = MotionLimits(velocity=0.3, acceleration=1.0, jerk=10.0)


@dataclass
class ExecSettings:
    steps_per_chunk: int = 8
    max_seconds: float = 120.0
    stop_file: str = "/tmp/yam_stop"
    monitor_gate: bool = False
    monitor_max_age_s: float = 20.0
    max_consecutive_holds: int = 60
    frame_max_age_s: float = 0.25

    def __post_init__(self) -> None:
        if not 1 <= int(self.steps_per_chunk) <= CHUNK_STEPS:
            raise ValueError(f"steps_per_chunk must be 1..{CHUNK_STEPS}")


class MonitorWorker:
    """Runs the Qwen/Astra pipeline off the control thread on the latest snapshot."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._snap: dict | None = None
        self._cv = threading.Condition()
        self._stop = False
        self.latest: Any = None
        self.latest_at: float | None = None
        self.errors = 0
        self._t = threading.Thread(target=self._loop, name="yam-monitor", daemon=True)

    def start(self) -> "MonitorWorker":
        self._t.start()
        return self

    def post(self, snap: dict) -> None:
        with self._cv:
            self._snap = snap
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._snap is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                snap, self._snap = self._snap, None
            try:
                out = self.pipeline.step(**snap)
                self.latest, self.latest_at = out, time.monotonic()
            except Exception:                                  # noqa: BLE001
                self.errors += 1

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()


@dataclass
class CycleRecord:
    cycle: int
    started_at: float
    outcome: str = ""
    reason: str = ""
    policy_state: list[float] | None = None
    proposal_first: list[float] | None = None
    pi05_s: float | None = None
    steps_decided: int | None = None
    decided_by: str | None = None
    steps_executed: int = 0
    violations_before: list[str] = field(default_factory=list)
    violations_after: list[str] = field(default_factory=list)
    sanitizer_changed: bool = False
    max_displacement_rad: float | None = None
    motion: dict[str, Any] | None = None
    total_s: float | None = None

    def to_log(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["schema"] = SCHEMA
        return d


class ExecutionRun:
    """One armed run. Owns nothing but the loop; the motion owner owns the arms."""

    def __init__(self, owner: Any, *, propose: Callable[[dict], dict],
                 grab_frames: Callable[[], dict], settings: ExecSettings,
                 config: dict[str, Any], task: str,
                 monitor: MonitorWorker | None = None,
                 audit_path: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if execution_blockers(config):
            raise MotionRefused("not armed: " + "; ".join(execution_blockers(config)))
        self.owner, self.propose, self.grab = owner, propose, grab_frames
        self.s, self.cfg, self.task = settings, config, task
        self.monitor = monitor
        self.audit_path = audit_path
        self.clock, self.sleep = clock, sleep
        self.max_disp = float(config["max_step_displacement"])
        self.records: list[CycleRecord] = []
        self._prev_frames: dict = {}
        self.stop_reason: str | None = None
        self.holds_in_a_row = 0

    # ----------------------------------------------------------------- state
    def policy_state(self) -> list[float]:
        """Measured arm joints, COMMANDED grippers (the reference)."""
        m = list(self.owner.measured)
        ref = list(self.owner.ref.position)
        for g in GRIPPER_INDICES.values():
            m[g] = ref[g]
        return m

    def _steps(self) -> tuple[int, str]:
        n = int(self.s.steps_per_chunk)
        if self.monitor is None:
            return n, "fixed"
        out, at = self.monitor.latest, self.monitor.latest_at
        if out is not None:
            d = (out.astra_decision or {}).get("decision") or out.astra_decision or {}
            if out.astra_called and str(d.get("mode")) == "stop":
                self.stop_reason = "Astra requested stop"
                return 0, "astra_stop"
        if not self.s.monitor_gate:
            return n, "fixed (monitor shadow)"
        if out is None or at is None or self.clock() - at > self.s.monitor_max_age_s:
            return 0, "no fresh monitor decision -> hold"
        return max(0, min(int(out.executed_steps), n)), f"gate:{out.controller}"

    # ----------------------------------------------------------------- cycle
    def cycle(self, n: int) -> CycleRecord:
        rec = CycleRecord(cycle=n, started_at=time.time())
        t0 = self.clock()
        try:
            self._cycle(rec)
        except MotionRefused as exc:
            rec.outcome, rec.reason = "refused", str(exc)[:300]
        except Exception as exc:                               # noqa: BLE001
            rec.outcome, rec.reason = "error", f"{type(exc).__name__}: {exc}"[:300]
        rec.total_s = round(self.clock() - t0, 4)
        rec.motion = self.owner.status()
        self.holds_in_a_row = 0 if rec.steps_executed else self.holds_in_a_row + 1
        self.records.append(rec)
        if self.audit_path:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_log(), default=str) + "\n")
        return rec

    def _hold(self, rec: CycleRecord, why: str) -> None:
        rec.outcome, rec.reason = "hold", why
        self.sleep(1.0 / CONTROL_HZ)

    def _cycle(self, rec: CycleRecord) -> None:
        state = self.policy_state()
        rec.policy_state = [round(v, 5) for v in state]
        frames = self.grab()
        labelled, meta, named = temporal_frames(frames, self._prev_frames)
        self._prev_frames = dict(frames)

        t = self.clock()
        prop = self.propose({"observation_id": f"exec:{rec.cycle}", "state": state,
                             "images": named, "task": self.task})
        rec.pi05_s = round(self.clock() - t, 4)
        if not prop.get("ok"):
            return self._hold(rec, f"pi0.5: {prop.get('error')}")
        rows = [list(r) for r in prop["rows"]]
        rec.proposal_first = [round(v, 5) for v in rows[0]]

        if self.monitor is not None:
            self.monitor.post(dict(
                state=state, proposed_steps=len(rows), frames=labelled or None,
                image_data_urls=[u for _, u in labelled] or None,
                frames_meta=meta or None, proposed_chunk=rows, task=self.task,
                episode_id="exec", state_text=state_text(state),
                intent_text=describe_trajectory(rows, state)))

        before = validate_chunk(rows, state=state)
        rec.violations_before = summarize(before)["codes"]
        if has_fatal(before):
            return self._hold(rec, "proposal has FATAL violations")

        steps, who = self._steps()
        rec.steps_decided, rec.decided_by = steps, who
        if steps == 0:
            return self._hold(rec, who)
        final, srep = sanitize(rows[:steps], state=state)
        rec.sanitizer_changed = bool(srep.changes_emitted)
        after = validate_chunk(final, state=state)
        rec.violations_after = summarize(after)["codes"]
        ok, blockers = execution_eligible(after, bool(final))
        if not ok:
            return self._hold(rec, "not execution-safe: " + ", ".join(blockers))
        disp = max(abs(r[j] - state[j]) for r in final for j in JOINT_INDICES)
        rec.max_displacement_rad = round(disp, 5)
        if disp > self.max_disp:
            return self._hold(rec, f"prefix moves {disp:.3f} rad > max_step_displacement "
                                   f"{self.max_disp}")

        start = self.clock()
        for i, row in enumerate(final):
            if self.stop_requested():
                break
            self.owner.set_goal(row)
            rec.steps_executed += 1
            wait = start + (i + 1) / CONTROL_HZ - self.clock()
            if wait > 0:
                self.sleep(wait)
        rec.outcome = "executed"
        rec.reason = f"{rec.steps_executed} step(s) ({who})"

    # ------------------------------------------------------------------- run
    def stop_requested(self) -> bool:
        from pathlib import Path
        if self.owner.fault:
            self.stop_reason = self.stop_reason or f"motion fault: {self.owner.fault}"
        elif Path(self.s.stop_file).exists():
            self.stop_reason = self.stop_reason or f"stop file {self.s.stop_file}"
        return self.stop_reason is not None

    def run(self, on_cycle: Callable[[CycleRecord], None] | None = None) -> dict[str, Any]:
        t0 = self.clock()
        n = 0
        while not self.stop_requested():
            if self.clock() - t0 > self.s.max_seconds:
                self.stop_reason = f"max_seconds {self.s.max_seconds} reached"
                break
            if self.holds_in_a_row >= self.s.max_consecutive_holds:
                self.stop_reason = f"{self.holds_in_a_row} holds in a row"
                break
            n += 1
            rec = self.cycle(n)
            if on_cycle:
                on_cycle(rec)
        return self.summary(self.clock() - t0)

    def summary(self, elapsed: float) -> dict[str, Any]:
        ex = [r for r in self.records if r.outcome == "executed"]
        by: dict[str, int] = {}
        for r in self.records:
            by[r.outcome] = by.get(r.outcome, 0) + 1
        lat = [r.pi05_s for r in self.records if r.pi05_s is not None]
        return {"schema": SCHEMA, "cycles": len(self.records), "outcomes": by,
                "steps_executed": sum(r.steps_executed for r in ex),
                "elapsed_s": round(elapsed, 2), "stop_reason": self.stop_reason,
                "pi05_mean_s": round(sum(lat) / len(lat), 3) if lat else None,
                "motion": self.owner.status(),
                "monitor_errors": self.monitor.errors if self.monitor else None}
