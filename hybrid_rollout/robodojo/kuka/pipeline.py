"""Policy routing: pi05_only, pi05_local_monitor, pi05_local_monitor_astra.

THIS IS A SECOND, ORTHOGONAL AXIS -- NOT A RENAME OF THE EXISTING MODES.
`safety.Mode` says how far a cycle may travel toward the hardware
(replay / live_shadow / reviewed_execution / astra_direct). `PolicyMode` here
says WHO decides. The two compose, and conflating them would silently change the
meaning of existing experiments, so the existing enum is untouched and
COMPAT_ALIASES records the mapping explicitly.

    pi05_only                  pi0.5 proposes; a fixed bounded prefix executes.
                               No monitor, no Astra. This is the existing
                               behaviour and its semantics are unchanged.
    pi05_local_monitor         a local VLM watches at ~2-5 Hz or on semantic
                               events and gates the prefix length. Never calls
                               Astra, never commands.
    pi05_local_monitor_astra   as above, plus escalation to Astra on PERSISTENT
                               evidence of failure or misalignment, then
                               hand-back after verified recovery.

THE POINT OF THE MIDDLE MODE
Astra is expensive and slow. Most cycles are unremarkable, and paying a large
model to say "still going fine" is waste. A 2B local model can say that cheaply
at a few Hz; the expensive reviewer is then reserved for the cases that actually
need judgement. The local monitor is a TRIAGE layer, which is why it may shorten
or stop but never correct.

SHADOW IS THE DEFAULT. In shadow the monitor records what it would have decided
and the executed prefix is unchanged. Gating requires an explicit reviewed
switch, and hardware gating defaults off regardless.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from .contract import ARM_DIM, MAX_STUDENT_STEPS
from .safety import Mode as ExecutionMode
from .vlm_backends import BackendProbe, MonitorBackend, UnconfiguredBackend
from .vlm_monitor import (Disposition, GateDecision, MonitorGate, MonitorPolicy,
                          MonitorRejected, MonitorReading)

SCHEMA = "hybrid_rollout.robodojo.kuka.pipeline.v1"


class PolicyMode(str, Enum):
    PI05_ONLY = "pi05_only"
    PI05_LOCAL_MONITOR = "pi05_local_monitor"
    PI05_LOCAL_MONITOR_ASTRA = "pi05_local_monitor_astra"


#: Existing names kept working. The mapping is documented rather than implied.
COMPAT_ALIASES: dict[str, PolicyMode] = {
    # what `cli run` and the experiment configs already call things
    "pi05_plus_astra": PolicyMode.PI05_LOCAL_MONITOR_ASTRA,
    "reviewed_execution": PolicyMode.PI05_LOCAL_MONITOR_ASTRA,
    "student_only": PolicyMode.PI05_ONLY,
    "pi05": PolicyMode.PI05_ONLY,
}

COMPAT_NOTE = (
    "pi05_plus_astra and reviewed_execution map to pi05_local_monitor_astra, "
    "which is a SUPERSET: the local monitor is added in front of the same Astra "
    "path. With the monitor in shadow (the default) the executed prefix is "
    "identical to before, so existing experiment semantics are preserved. "
    "Astra-direct is a different axis entirely and is not aliased here.")


def resolve_mode(name: str | PolicyMode) -> PolicyMode:
    if isinstance(name, PolicyMode):
        return name
    key = str(name).strip()
    try:
        return PolicyMode(key)
    except ValueError:
        pass
    if key in COMPAT_ALIASES:
        return COMPAT_ALIASES[key]
    raise ValueError(
        f"unknown policy mode {name!r}; no implicit fallback. Known: "
        f"{[m.value for m in PolicyMode]} plus aliases {sorted(COMPAT_ALIASES)}")


#: Semantic events worth a monitor look, beyond the periodic tick.
class Event(str, Enum):
    PERIODIC = "periodic"
    APPROACH_REGION = "approach_region_entered"
    GRIPPER_CLOSE = "gripper_close_commanded"
    GRIPPER_OPEN = "gripper_open_commanded"
    EXPECTED_LIFT = "expected_lift"
    EXPECTED_PLACEMENT = "expected_placement"
    STALLED_MOTION = "stalled_or_repeated_motion"


@dataclass(frozen=True)
class MonitorSchedule:
    """~2-5 Hz, or on a semantic event. Never every 250 Hz control tick."""
    min_interval_s: float = 0.20        # 5 Hz ceiling
    max_interval_s: float = 0.50        # 2 Hz floor
    stall_epsilon_deg: float = 0.05     # below this, motion counts as stalled
    stall_cycles: int = 8

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"schema": SCHEMA, "hz_range": [1.0 / self.max_interval_s,
                                                 1.0 / self.min_interval_s]})
        return d


class EventDetector:
    """Deterministic. Decides WHEN to look; never what to conclude."""

    def __init__(self, schedule: MonitorSchedule | None = None) -> None:
        self.sched = schedule or MonitorSchedule()
        self.last_eval: float | None = None
        self.last_state: list[float] | None = None
        self.still_cycles = 0
        self.last_gripper: float | None = None

    def due(self, *, state: Sequence[float], now: float | None = None,
            in_approach_region: bool = False) -> tuple[bool, Event | None]:
        now = time.time() if now is None else now
        s = [float(v) for v in list(state)[:ARM_DIM]]
        grip = float(state[ARM_DIM]) if len(state) > ARM_DIM else None

        event: Event | None = None
        if self.last_state is not None:
            moved = max(abs(s[j] - self.last_state[j]) for j in range(ARM_DIM))
            if moved < self.sched.stall_epsilon_deg:
                self.still_cycles += 1
            else:
                self.still_cycles = 0
            if self.still_cycles >= self.sched.stall_cycles:
                event = Event.STALLED_MOTION
        if grip is not None and self.last_gripper is not None:
            if grip >= 0.5 > self.last_gripper:
                event = Event.GRIPPER_CLOSE
            elif grip < 0.5 <= self.last_gripper:
                event = Event.GRIPPER_OPEN
        if in_approach_region and event is None:
            event = Event.APPROACH_REGION

        self.last_state, self.last_gripper = s, grip
        elapsed = None if self.last_eval is None else now - self.last_eval
        if event is not None and (elapsed is None
                                  or elapsed >= self.sched.min_interval_s):
            self.last_eval = now
            return True, event
        if elapsed is None or elapsed >= self.sched.max_interval_s:
            self.last_eval = now
            return True, Event.PERIODIC
        return False, None


@dataclass
class CycleOutcome:
    """One routed cycle. Append-only audit row."""
    mode: str
    cycle: int
    event: str | None
    proposed_steps: int
    executed_steps: int
    baseline_steps: int              # what pi05_only would have executed
    gate: dict[str, Any] | None
    monitor_error: str | None
    escalated: bool
    astra_called: bool
    astra_decision: dict[str, Any] | None
    handed_back: bool
    controller: str                  # pi05 | astra | none
    shadow: bool
    reason: str
    episode_id: str | None = None
    task: str | None = None
    monitor_latency_s: float | None = None
    decided_at: float = field(default_factory=time.time)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"schema": SCHEMA, "authoritative": False,
                  "note": ("advisory routing; Ruckig, the execution state "
                           "machine, the RSI gateway, joint/velocity/"
                           "acceleration/jerk limits, the staleness watchdog, "
                           "operator stop and the physical E-stop remain "
                           "authoritative and are not bypassed")})
        return d


class PolicyPipeline:
    """Routes a cycle according to PolicyMode. Commands nothing itself."""

    def __init__(self, mode: str | PolicyMode = PolicyMode.PI05_ONLY, *,
                 backend: MonitorBackend | None = None,
                 policy: MonitorPolicy | None = None,
                 schedule: MonitorSchedule | None = None,
                 shadow: bool = True,
                 astra_review: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 default_steps: int = 5,
                 on_record: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.mode = resolve_mode(mode)
        self.backend = backend or UnconfiguredBackend()
        self.gate = MonitorGate(policy, shadow=shadow)
        self.events = EventDetector(schedule)
        self.shadow = bool(shadow)
        self.astra_review = astra_review
        self.default_steps = max(1, min(default_steps, MAX_STUDENT_STEPS))
        self.on_record = on_record
        self.cycle = 0
        self.controller = "pi05"
        self.records: list[CycleOutcome] = []

    # ------------------------------------------------------------- readiness
    def preflight(self) -> tuple[bool, list[str]]:
        """Is this mode runnable? A missing monitor service blocks gating."""
        blockers: list[str] = []
        if self.mode is PolicyMode.PI05_ONLY:
            return True, []
        probe: BackendProbe = self.backend.probe()
        if not probe.available:
            blockers.append(f"monitor backend unavailable: {probe.reason}")
        if self.mode is PolicyMode.PI05_LOCAL_MONITOR_ASTRA and self.astra_review is None:
            blockers.append("no Astra reviewer configured for the escalation path")
        return (not blockers), blockers

    # ----------------------------------------------------------------- cycle
    def step(self, *, state: Sequence[float], proposed_steps: int,
             frames: Sequence[str] | None = None, state_text: str = "",
             intent_text: str = "", in_approach_region: bool = False,
             episode_id: str | None = None, task: str | None = None,
             now: float | None = None) -> CycleOutcome:
        now = time.time() if now is None else now
        self.cycle += 1
        baseline = max(0, min(int(proposed_steps), self.default_steps))

        if self.mode is PolicyMode.PI05_ONLY:
            return self._record(CycleOutcome(
                self.mode.value, self.cycle, None, proposed_steps, baseline,
                baseline, None, None, False, False, None, False, "pi05",
                self.shadow, "pi05_only: fixed bounded prefix, no monitor",
                episode_id, task))

        due, event = self.events.due(state=state, now=now,
                                     in_approach_region=in_approach_region)
        if not due:
            return self._record(CycleOutcome(
                self.mode.value, self.cycle, None, proposed_steps, baseline,
                baseline, None, None, False, False, None, False, self.controller,
                self.shadow, "monitor not due this cycle; prior decision stands",
                episode_id, task))

        reading: MonitorReading | None = None
        err: str | None = None
        try:
            reading = self.backend.observe(
                frames=list(frames or []), state_text=state_text,
                intent_text=intent_text, timeout_s=None)
        except MonitorRejected as exc:
            err = f"{exc.code}: {exc.detail}"
        except Exception as exc:                                   # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"[:180]

        decision: GateDecision = self.gate.evaluate(
            reading, proposed_steps=proposed_steps, now=now, error=err)

        escalate = (decision.disposition is Disposition.ESCALATE
                    and self.mode is PolicyMode.PI05_LOCAL_MONITOR_ASTRA)
        astra_out: dict[str, Any] | None = None
        handed_back = False

        if escalate and self.astra_review is not None:
            self.controller = "astra"
            try:
                astra_out = self.astra_review({
                    "cycle": self.cycle, "reason": decision.reason,
                    "reading": decision.reading, "state": list(state),
                    "episode_id": episode_id, "task": task})
            except Exception as exc:                               # noqa: BLE001
                astra_out = {"ok": False,
                             "error": f"{type(exc).__name__}: {exc}"[:180]}
        elif self.controller == "astra":
            ok, why = self.gate.may_hand_back()
            if ok:
                self.gate.hand_back()
                self.controller = "pi05"
                handed_back = True

        # SHADOW: record the decision, execute the unchanged baseline.
        executed = baseline if self.shadow else decision.execute_steps
        reason = decision.reason
        if self.shadow and executed != decision.execute_steps:
            reason = (f"SHADOW: would have executed {decision.execute_steps}, "
                      f"actually executed the unchanged baseline {executed}. "
                      f"{decision.reason}")

        return self._record(CycleOutcome(
            self.mode.value, self.cycle, event.value if event else None,
            proposed_steps, executed, baseline, decision.to_log(), err,
            escalate, astra_out is not None, astra_out, handed_back,
            self.controller, self.shadow, reason, episode_id, task,
            reading.latency_s if reading else None))

    def _record(self, out: CycleOutcome) -> CycleOutcome:
        self.records.append(out)
        if self.on_record:
            self.on_record(out.to_log())
        return out

    # --------------------------------------------------------------- metrics
    def metrics(self) -> dict[str, Any]:
        by_controller: dict[str, int] = {}
        for r in self.records:
            by_controller[r.controller] = by_controller.get(r.controller, 0) + 1
        lat = [r.monitor_latency_s for r in self.records
               if r.monitor_latency_s is not None]
        rejected = [r for r in self.records if r.monitor_error]
        return {
            "schema": SCHEMA, "mode": self.mode.value, "shadow": self.shadow,
            "cycles": self.cycle,
            "cycles_by_controller": by_controller,
            "escalations": sum(1 for r in self.records if r.escalated),
            "astra_calls": sum(1 for r in self.records if r.astra_called),
            "hand_backs": sum(1 for r in self.records if r.handed_back),
            "rejected_monitor_outputs": len(rejected),
            "rejection_reasons": sorted({r.monitor_error for r in rejected
                                         if r.monitor_error})[:8],
            "steps_executed": sum(r.executed_steps for r in self.records),
            "steps_baseline": sum(r.baseline_steps for r in self.records),
            "monitor_latency_s": {
                "n": len(lat),
                "mean": round(sum(lat) / len(lat), 4) if lat else None,
                "max": round(max(lat), 4) if lat else None},
            "gate": self.gate.metrics(),
            "schedule": self.events.sched.to_log(),
            "compat_aliases": {k: v.value for k, v in COMPAT_ALIASES.items()},
            "compat_note": COMPAT_NOTE,
        }
