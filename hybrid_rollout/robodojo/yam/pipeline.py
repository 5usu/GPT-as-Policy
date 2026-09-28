"""Policy routing on YAM: pi05_only, pi05_local_monitor, pi05_local_monitor_astra.

The KUKA branch's three-model architecture, unchanged in substance:

    pi0.5 proposes a chunk  ->  Qwen3-VL-2B on the Jetson triages it  ->
    Astra is asked ONLY on persistent adverse evidence  ->  hand-back to pi0.5
    after verified recovery

Everything that decides is imported from the KUKA package and not copied: the
PolicyMode enum and its aliases, the MonitorGate with its streaks, clamps and
fail-safe, the monitor backends and prompt, CycleOutcome, and the escalation
/ re-ask / hand-back logic in `PolicyPipeline.step`. Two pieces had to change
because they know the shape of a KUKA row:

  EventDetector   stall is measured over TWELVE joints in radians, and a
                  gripper event fires per arm with YAM polarity -- dropping
                  below 0.5 is CLOSING here, the opposite of the KUKA cell.
  build_packet    an escalation forwards a bimanual packet (yam.packet), not
                  a single-arm KUKA one.

The packet is swapped by overriding `_build_escalation_packet`, which the KUKA
class does not have -- so `step` is re-stated here with that single call
changed. Keep the two in sync; test_yam_pipeline pins the shared behaviour.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..kuka.pipeline import (COMPAT_ALIASES, COMPAT_NOTE, CycleOutcome, Event,
                             MonitorSchedule, PolicyMode, PolicyPipeline,
                             resolve_mode)
from ..kuka.vlm_backends import MonitorBackend
from ..kuka.vlm_monitor import (Disposition, GateDecision, MonitorPolicy,
                                MonitorReading, MonitorRejected)
from .contract import (ARMS, CHUNK_STEPS, CONTROL_HZ, GRIPPER_INDICES,
                       GRIPPER_THRESHOLD, JOINT_INDICES, split_arms)
from .packet import build_packet

SCHEMA = "hybrid_rollout.robodojo.yam.pipeline.v1"
INTENT_PREVIEW_STEPS = 5

__all__ = ["COMPAT_ALIASES", "COMPAT_NOTE", "Event", "PolicyMode",
           "YamEventDetector", "YamMonitorSchedule", "YamPolicyPipeline",
           "describe_trajectory", "resolve_mode", "state_text"]


def _r(values: Sequence[float], nd: int = 3) -> list[float]:
    return [round(float(v), nd) for v in values]


def state_text(state: Sequence[float]) -> str:
    """Deterministic state line for the monitor."""
    a = split_arms(state)
    return " | ".join(f"{arm} joints_rad={_r(a[arm]['joints'])} "
                      f"gripper={float(a[arm]['gripper']):.3f} (0 closed, 1 open)"
                      for arm in ARMS)


def describe_trajectory(chunk: Sequence[Sequence[float]] | None,
                        state: Sequence[float] | None = None,
                        *, preview: int = INTENT_PREVIEW_STEPS) -> str:
    """The ACTUAL proposed targets, per arm, for the monitor."""
    if not chunk:
        return ("NO PROPOSED TRAJECTORY SUPPLIED. Report intent as uncertain; "
                "do not infer one.")
    rows = [list(r) for r in chunk]
    n = len(rows)
    lines = [f"{n} absolute joint targets in radians at {CONTROL_HZ:.0f} Hz "
             f"({n / CONTROL_HZ:.2f} s) for two arms; grippers 0 closed, 1 open.",
             f"first {min(preview, n)} shown verbatim:"]
    for i, r in enumerate(rows[:preview]):
        a = split_arms(r)
        lines.append(f"  t+{i:02d}: L {_r(a['left']['joints'], 2)} "
                     f"g={float(a['left']['gripper']):.2f} | "
                     f"R {_r(a['right']['joints'], 2)} "
                     f"g={float(a['right']['gripper']):.2f}")
    if state is not None:
        s, e = split_arms(state), split_arms(rows[-1])
        for arm in ARMS:
            net = [round(float(x) - float(y), 3)
                   for x, y in zip(e[arm]["joints"], s[arm]["joints"])]
            lines.append(f"{arm} net joint displacement over the chunk: {net} rad")
    for arm, g in GRIPPER_INDICES.items():
        g0, g1 = float(rows[0][g]), float(rows[-1][g])
        if abs(g1 - g0) > 0.05:
            lines.append(f"{arm} gripper {g0:.2f} -> {g1:.2f} "
                         f"({'closing' if g1 < g0 else 'opening'})")
        else:
            lines.append(f"{arm} gripper held near {g0:.2f}")
    return "\n".join(lines)


@dataclass(frozen=True)
class YamMonitorSchedule(MonitorSchedule):
    """KUKA schedule; `stall_epsilon_rad` replaces the degree threshold and
    the 'cycles between looks' figure uses the 30 Hz chunk clock."""
    stall_epsilon_rad: float = 0.001

    @classmethod
    def from_latency(cls, latency_s: float, **kw) -> "YamMonitorSchedule":
        return cls(min_interval_s=max(0.2, latency_s),
                   max_interval_s=max(1.0, latency_s * 2.0),
                   measured_latency_s=latency_s, **kw)

    def to_log(self) -> dict[str, Any]:
        d = super().to_log()
        d.pop("stall_epsilon_deg", None)
        d["schema"] = SCHEMA
        d["control_cycles_between_looks"] = int(
            (self.measured_latency_s or self.max_interval_s) * CONTROL_HZ)
        return d


_SEVERITY = {Event.PERIODIC: 0, Event.APPROACH_REGION: 1,
             Event.STALLED_MOTION: 2, Event.GRIPPER_OPEN: 3,
             Event.GRIPPER_CLOSE: 4}


class YamEventDetector:
    """Deterministic. Decides WHEN to look; never what to conclude.

    Same semantics as `kuka.pipeline.EventDetector` -- held-not-dropped events
    while rate-limited, periodic looks at max_interval_s -- over both arms.
    """

    def __init__(self, schedule: MonitorSchedule | None = None) -> None:
        self.sched = schedule or YamMonitorSchedule()
        self.eps = float(getattr(self.sched, "stall_epsilon_rad", 0.001))
        self.last_eval: float | None = None
        self.last_state: list[float] | None = None
        self.still_cycles = 0
        self.last_grip: dict[str, float] | None = None
        self.pending: Event | None = None
        self.deferred_events = 0
        self.last_event_arm: str | None = None

    def due(self, *, state: Sequence[float], now: float | None = None,
            in_approach_region: bool = False) -> tuple[bool, Event | None]:
        now = time.time() if now is None else now
        s = [float(v) for v in state]
        grip = {arm: s[g] for arm, g in GRIPPER_INDICES.items()} if len(s) > 13 else None

        event: Event | None = None
        if self.last_state is not None:
            moved = max(abs(s[j] - self.last_state[j]) for j in JOINT_INDICES)
            self.still_cycles = self.still_cycles + 1 if moved < self.eps else 0
            if self.still_cycles >= self.sched.stall_cycles:
                event = Event.STALLED_MOTION
        if grip is not None and self.last_grip is not None:
            for arm in ARMS:
                was, now_g = self.last_grip[arm], grip[arm]
                if now_g < GRIPPER_THRESHOLD <= was:
                    event, self.last_event_arm = Event.GRIPPER_CLOSE, arm
                elif was < GRIPPER_THRESHOLD <= now_g and event is not Event.GRIPPER_CLOSE:
                    event, self.last_event_arm = Event.GRIPPER_OPEN, arm
        if in_approach_region and event is None:
            event = Event.APPROACH_REGION

        self.last_state, self.last_grip = s, grip
        elapsed = None if self.last_eval is None else now - self.last_eval
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


class YamPolicyPipeline(PolicyPipeline):
    """`kuka.pipeline.PolicyPipeline` with the YAM detector and packet."""

    def __init__(self, mode: str | PolicyMode = PolicyMode.PI05_ONLY, *,
                 backend: MonitorBackend | None = None,
                 policy: MonitorPolicy | None = None,
                 schedule: MonitorSchedule | None = None,
                 shadow: bool = True,
                 astra_review: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 default_steps: int = 5,
                 task_reference: Any = None,
                 on_record: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__(mode, backend=backend, policy=policy, schedule=schedule,
                         shadow=shadow, astra_review=astra_review,
                         default_steps=default_steps, task_reference=task_reference,
                         on_record=on_record)
        self.events = YamEventDetector(schedule)
        self.gate.chunk_steps = CHUNK_STEPS

    def _build_escalation_packet(self, *, state, proposed_chunk, frames_meta,
                                 fk_preview, episode_id, task) -> dict[str, Any]:
        return build_packet(
            task_instruction=task or "",
            observation_id=f"{episode_id or 'ep'}:c{self.cycle:06d}",
            state=list(state), chunk=proposed_chunk or [list(state)],
            provenance="model_predicted", frames=frames_meta or {},
            fk_preview=fk_preview, reference=self.task_reference)

    def step(self, *, state: Sequence[float], proposed_steps: int,
             frames: Sequence[Any] | None = None, state_text: str = "",
             intent_text: str = "", in_approach_region: bool = False,
             episode_id: str | None = None, task: str | None = None,
             now: float | None = None,
             proposed_chunk: Sequence[Sequence[float]] | None = None,
             frames_meta: dict[str, Any] | None = None,
             fk_preview: dict[str, Any] | None = None,
             image_data_urls: Sequence[str] | None = None) -> CycleOutcome:
        # Mirrors kuka.pipeline.PolicyPipeline.step; only the packet differs.
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

        already_escalated = self.controller == "astra"
        recall = self.gate.policy.astra_recall_every
        due_recall = (already_escalated and recall > 0
                      and (self.cycle - self._escalated_at) % recall == 0)
        ask_astra = (escalate and self.astra_review is not None
                     and (not already_escalated or due_recall))
        reason_suffix = ""
        if escalate and not ask_astra and already_escalated:
            reason_suffix = (f" (Astra already holds control since cycle "
                             f"{self._escalated_at}; not re-asked)")

        if ask_astra:
            if not already_escalated:
                self._escalated_at = self.cycle
            self.controller = "astra"
            try:
                pkt = self._build_escalation_packet(
                    state=state, proposed_chunk=proposed_chunk,
                    frames_meta=frames_meta, fk_preview=fk_preview,
                    episode_id=episode_id, task=task)
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
            astra_steps = self._last_astra_steps
        elif self.controller == "astra":
            ok, _why = self.gate.may_hand_back()
            if ok:
                self.gate.hand_back()
                self.controller = "pi05"
                handed_back = True

        gated = decision.execute_steps if astra_steps is None else astra_steps
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

    def metrics(self) -> dict[str, Any]:
        m = super().metrics()
        m["schema"] = SCHEMA
        m["robot"] = "yam_bimanual"
        return m
