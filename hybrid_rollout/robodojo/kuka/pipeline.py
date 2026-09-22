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
    pi05_local_monitor         a local VLM watches at its MEASURED rate (a 2B
                               on an AGX Orin is ~0.2 Hz, not 2-5) or on semantic
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
from .packet import build_packet
from .safety import Mode as ExecutionMode
from .vlm_backends import BackendProbe, MonitorBackend, UnconfiguredBackend
from .vlm_monitor import (Disposition, GateDecision, MonitorGate, MonitorPolicy,
                          MonitorRejected, MonitorReading)

SCHEMA = "hybrid_rollout.robodojo.kuka.pipeline.v1"

#: How many leading targets to show the monitor verbatim. A 500M model's
#: context is small and 50x7 floats would swamp the images; the leading rows
#: plus per-joint net displacement carry the signal that matters for "is this
#: heading where it should".
INTENT_PREVIEW_STEPS = 5


def describe_trajectory(chunk: Sequence[Sequence[float]] | None,
                        state: Sequence[float] | None = None,
                        *, preview: int = INTENT_PREVIEW_STEPS) -> str:
    """Render the ACTUAL proposed targets for the monitor.

    The first version sent only a phrase -- "next 50 absolute joint targets" --
    the COUNT and not the values, then asked whether the intent was aligned.
    There was nothing to align against, so uncertain was the only honest answer.
    """
    if not chunk:
        return ("NO PROPOSED TRAJECTORY SUPPLIED. Report intent as uncertain; "
                "do not infer one.")
    rows = [list(r) for r in chunk]
    n = len(rows)
    lines = [f"{n} absolute joint targets in degrees at 30 Hz "
             f"({n / 30.0:.2f} s), A1-A6 then gripper.",
             f"first {min(preview, n)} shown verbatim:"]
    for i, r in enumerate(rows[:preview]):
        lines.append(f"  t+{i:02d}: {[round(float(v), 2) for v in r[:ARM_DIM]]} "
                     f"grip={float(r[ARM_DIM]):.2f}" if len(r) > ARM_DIM
                     else f"  t+{i:02d}: {[round(float(v), 2) for v in r[:ARM_DIM]]}")
    if state is not None and len(state) >= ARM_DIM:
        net = [round(float(rows[-1][j]) - float(state[j]), 2)
               for j in range(ARM_DIM)]
        lines.append(f"net displacement from the current pose over the whole "
                     f"chunk: {net} deg")
    if len(rows[0]) > ARM_DIM:
        g0, g1 = float(rows[0][ARM_DIM]), float(rows[-1][ARM_DIM])
        if abs(g1 - g0) > 0.05:
            lines.append(f"gripper transitions {g0:.2f} -> {g1:.2f} "
                         f"({'closing' if g1 > g0 else 'opening'})")
        else:
            lines.append(f"gripper held near {g0:.2f}")
    return "\n".join(lines)


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
    """When to look. Never every 250 Hz control tick.

    THE RATE IS BOUNDED BY INFERENCE, NOT BY THIS CLASS. Asking every 0.2 s is
    meaningless if a reading takes 6 s -- the schedule would simply request the
    next look before the last one returned, and the effective rate would be
    whatever the model manages. The defaults below are therefore set from
    MEASURED device latency, and `effective_hz` reports what is actually
    achievable so a run cannot quietly believe it is monitoring five times a
    second when it is monitoring once every six.

    The safety consequence is real and should be stated rather than buried: a
    slower monitor reacts later. At ~6 s a check, roughly 180 control cycles
    pass between looks, which is why the deterministic stop conditions -- limits,
    the staleness watchdog, commanded-vs-measured -- remain the things that
    actually protect the arm. The monitor is triage, not a safety layer.
    """
    min_interval_s: float = 1.0
    max_interval_s: float = 6.0
    stall_epsilon_deg: float = 0.05     # below this, motion counts as stalled
    stall_cycles: int = 8
    #: Measured seconds per reading on the target device. None = unmeasured.
    measured_latency_s: float | None = None

    @classmethod
    def from_latency(cls, latency_s: float, **kw) -> "MonitorSchedule":
        """Build a schedule that admits what the device can actually do."""
        return cls(min_interval_s=max(0.2, latency_s),
                   max_interval_s=max(1.0, latency_s * 2.0),
                   measured_latency_s=latency_s, **kw)

    def effective_hz(self) -> float:
        """What is achievable, not what is requested."""
        floor = max(self.min_interval_s, self.measured_latency_s or 0.0)
        return 1.0 / floor if floor else 0.0

    def honest(self) -> tuple[bool, str]:
        if self.measured_latency_s is None:
            return False, ("monitor latency has not been measured on this "
                           "device; the requested rate is an assumption")
        if self.measured_latency_s > self.min_interval_s + 1e-9:
            return False, (f"requested up to {1.0/self.min_interval_s:.1f} Hz but "
                           f"a reading takes {self.measured_latency_s:.2f}s, so "
                           f"the real rate is {self.effective_hz():.2f} Hz")
        return True, "requested rate is achievable at the measured latency"

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        ok, why = self.honest()
        d.update({"schema": SCHEMA,
                  "requested_hz_range": [
                      (1.0 / self.max_interval_s) if self.max_interval_s else None,
                      (1.0 / self.min_interval_s) if self.min_interval_s else None],
                  "effective_hz": round(self.effective_hz(), 3),
                  "rate_is_honest": ok, "rate_note": why,
                  "control_cycles_between_looks": int(
                      (self.measured_latency_s or self.max_interval_s) * 250)})
        return d


# Which event wins when several occur inside one rate-limited window.
_SEVERITY = {Event.PERIODIC: 0, Event.APPROACH_REGION: 1,
             Event.STALLED_MOTION: 2, Event.GRIPPER_OPEN: 3,
             Event.GRIPPER_CLOSE: 4}


class EventDetector:
    """Deterministic. Decides WHEN to look; never what to conclude."""

    def __init__(self, schedule: MonitorSchedule | None = None) -> None:
        self.sched = schedule or MonitorSchedule()
        self.last_eval: float | None = None
        self.last_state: list[float] | None = None
        self.still_cycles = 0
        self.last_gripper: float | None = None
        self.pending: Event | None = None   # semantic event seen while rate-limited
        self.deferred_events = 0            # how often that happened (reported)

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

        # A semantic event that arrives while we are rate-limited is HELD, not
        # dropped. The monitor cannot be re-entered faster than one inference
        # (~6 s on the Jetson), but "the model is busy" must never become "the
        # gripper closed and nobody looked" -- the gripper is driven over Modbus
        # straight from the Jetson, so RSI STOPFLAG does not stop it. The event
        # survives to the next look; only its TIMELINESS degrades, and that is
        # counted so the delay is visible rather than silent.
        if event is not None:
            if self.pending is None or _SEVERITY[event] > _SEVERITY[self.pending]:
                self.pending = event

        ready = elapsed is None or elapsed >= self.sched.min_interval_s
        if self.pending is not None and ready:
            held, self.pending = self.pending, None
            self.last_eval = now
            return True, held
        if self.pending is not None:
            self.deferred_events += 1
            return False, None
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
        self._escalated_at = 0
        self._last_astra_steps: int | None = None
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
             now: float | None = None,
             proposed_chunk: Sequence[Sequence[float]] | None = None,
             frames_meta: dict[str, Any] | None = None,
             fk_preview: dict[str, Any] | None = None,
             image_data_urls: Sequence[str] | None = None) -> CycleOutcome:
        """`proposed_chunk`, `frames_meta`, `fk_preview` and `image_data_urls`
        are what an escalation forwards to Astra. Without them the reviewer
        would be judging the scene from the monitor's prose."""
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
        astra_steps: int | None = None
        handed_back = False

        # Astra is asked on the TRANSITION into escalation, and thereafter only
        # on the configured cadence. Without this it was re-called on every
        # adverse tick for as long as the condition lasted.
        already_escalated = self.controller == "astra"
        recall = self.gate.policy.astra_recall_every
        due_recall = (already_escalated and recall > 0
                      and (self.cycle - self._escalated_at) % recall == 0)
        ask_astra = (escalate and self.astra_review is not None
                     and (not already_escalated or due_recall))
        if escalate and not ask_astra and already_escalated:
            reason_suffix = (f" (Astra already holds control since cycle "
                             f"{self._escalated_at}; not re-asked)")
        else:
            reason_suffix = ""

        if ask_astra:
            if not already_escalated:
                self._escalated_at = self.cycle
            self.controller = "astra"
            # A REVIEW PACKET, not a summary of the monitor's opinion.
            # Astra has to see what the monitor saw -- the frames and the
            # proposed trajectory -- or it is being asked to judge a scene from
            # prose. The shape is `packet.build_packet`, the same one the Astra
            # client consumes, so a mismatch is a TypeError here rather than a
            # KeyError swallowed downstream.
            try:
                pkt = build_packet(
                    task_instruction=task or "",
                    observation_id=f"{episode_id or 'ep'}:c{self.cycle:06d}",
                    state=list(state), chunk=proposed_chunk or [list(state)],
                    provenance="model_predicted", frames=frames_meta or {},
                    fk_preview=fk_preview)
                pkt["escalation"] = {
                    "cause": decision.reason,
                    "monitor_reading": decision.reading,
                    "adverse_streak": decision.adverse_streak,
                    "uncertain_streak": decision.uncertain_streak}
                if image_data_urls:
                    pkt["image_data_urls"] = list(image_data_urls)
                astra_out = self.astra_review(pkt)
            except Exception as exc:                               # noqa: BLE001
                astra_out = {"ok": False,
                             "error": f"{type(exc).__name__}: {exc}"[:180]}
            astra_steps = self._astra_steps(astra_out, proposed_steps)
            self._last_astra_steps = astra_steps
        elif escalate and already_escalated:
            # Still escalated, not re-asking. Astra's standing decision holds.
            astra_steps = self._last_astra_steps
        elif self.controller == "astra":
            ok, why = self.gate.may_hand_back()
            if ok:
                self.gate.hand_back()
                self.controller = "pi05"
                handed_back = True

        # Who decides how many steps run?
        #   shadow           -> nobody; the unchanged baseline executes
        #   astra in control -> Astra's clamped answer, falling back to the
        #                       monitor's number when it gave nothing usable
        #   otherwise        -> the monitor gate
        gated = decision.execute_steps
        if astra_steps is not None:
            gated = astra_steps
        executed = baseline if self.shadow else gated
        reason = decision.reason
        if astra_steps is not None:
            reason = (f"astra reviewed and governs this cycle: {astra_steps} "
                      f"step(s). {decision.reason}")
        elif escalate and astra_out is not None:
            reason = (f"ESCALATED but Astra returned nothing usable "
                      f"({astra_out.get('error') or 'no decision'}); falling "
                      f"back to the monitor's {gated} step(s). {decision.reason}")
        if self.shadow and executed != decision.execute_steps:
            reason = (f"SHADOW: would have executed {decision.execute_steps}, "
                      f"actually executed the unchanged baseline {executed}. "
                      f"{decision.reason}")

        return self._record(CycleOutcome(
            self.mode.value, self.cycle, event.value if event else None,
            proposed_steps, executed, baseline, decision.to_log(), err,
            escalate, ask_astra, astra_out, handed_back,
            self.controller, self.shadow, reason + reason_suffix, episode_id, task,
            reading.latency_s if reading else None))

    def _astra_steps(self, astra_out: dict[str, Any] | None,
                     proposed_steps: int) -> int | None:
        """Astra's answer GOVERNS while it holds control -- but is clamped.

        Returning None means Astra gave nothing usable, and the caller must fall
        back to the monitor's (already conservative) number rather than assume
        permission.
        """
        if not astra_out or not astra_out.get("ok"):
            return None
        d = astra_out.get("decision") or astra_out
        mode = str(d.get("mode", ""))
        if mode == "stop":
            return 0
        if mode not in ("student", "edit", "astra_direct_joint"):
            return None
        steps = d.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            return None
        # Same ceilings as the monitor path. A reviewer is not exempt.
        return max(0, min(steps, MAX_STUDENT_STEPS, int(proposed_steps),
                          self.gate.chunk_steps))

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
            # Events that occurred while the monitor was mid-inference. They
            # were delivered late, not dropped -- but a large number here means
            # the model is too slow for the motion it is watching.
            "events_deferred_by_monitor_latency": self.events.deferred_events,
            "schedule": self.events.sched.to_log(),
            "compat_aliases": {k: v.value for k, v in COMPAT_ALIASES.items()},
            "compat_note": COMPAT_NOTE,
        }
