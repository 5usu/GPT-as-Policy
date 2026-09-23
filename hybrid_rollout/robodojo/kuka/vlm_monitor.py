"""Local-VLM monitor: an event/status observer and escalation gate. NOT a controller.

WHERE THIS SITS
    pi0.5 proposes a 50-step chunk   (fast, the primary policy)
      -> this monitor watches recent frames + deterministic state + the intended
         trajectory, paced by inference latency measured ON THE DEVICE (the
         board is unidentified; no rate here is a specification)
      -> it returns a STATUS, and a bounded number of steps it is willing to see
         executed, and whether to escalate
      -> Astra is called only on persistent evidence of failure or misalignment
      -> Ruckig / the execution state machine / RSI gateway remain authoritative

IT NEVER TOUCHES THE 250 Hz LOOP. Inference runs out of band; the RSI cycle reads
a cached decision and never waits on a model.

THREE RULES THAT MAKE THIS SAFE

1. `execute_steps` IS CLAMPED INDEPENDENTLY OF THE MODEL. Whatever number comes
   back is bounded by `MonitorPolicy` limits derived from the phase and the
   deterministic state. A model that returns 999 gets the same ceiling as one
   that returns 3. Confidence is ADVISORY and never widens a bound.

2. ONE FRAME CANNOT CAUSE A TAKEOVER. Escalation requires the same adverse
   condition to persist across N consecutive evaluations. A single noisy
   observation shortens the prefix; it does not hand control away.

3. ABSENCE OF AN ANSWER IS THE MOST CONSERVATIVE ANSWER. Stale, malformed,
   timed-out or unavailable responses do not mean "carry on" -- they collapse
   `execute_steps` to zero, which in the existing state machine means HOLD:
   keep answering the controller with the measured pose and STOPFLAG=1, command
   nothing. Silence is never treated as consent.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.vlm_monitor.v1"

#: Evidence is capped tightly because output tokens are the latency. See the
#: schema comment; this is a speed decision, not a style one.
EVIDENCE_MAX_CHARS = 120


class Progress(str, Enum):
    NORMAL = "normal"
    STALLED = "stalled"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class Intent(str, Enum):
    ALIGNED = "aligned"
    UNCERTAIN = "uncertain"
    MISALIGNED = "misaligned"


class Phase(str, Enum):
    APPROACH = "approach"
    ALIGN = "align"
    CONTACT = "contact"
    GRASP = "grasp"
    MANIPULATE = "manipulate"
    RETREAT = "retreat"
    UNKNOWN = "unknown"


class Disposition(str, Enum):
    """What the deterministic layer decided, after reading the monitor."""
    ALLOW_PREFIX = "allow_prefix"     # execute a bounded prefix
    SHORTEN = "shorten"               # allow fewer steps than proposed
    HOLD_REOBSERVE = "hold_reobserve" # execute nothing; look again
    ESCALATE = "escalate"             # persistent adverse evidence -> Astra
    FAIL_SAFE = "fail_safe"           # no usable answer -> HOLD


#: Adverse conditions. Escalation requires PERSISTENCE across evaluations.
ADVERSE_PROGRESS = frozenset({Progress.STALLED, Progress.FAILED})
ADVERSE_INTENT = frozenset({Intent.MISALIGNED})


def response_schema() -> dict[str, Any]:
    """Strict, validated. Unknown fields are rejected rather than ignored."""
    s = {"type": "string"}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["phase", "progress", "target_visible", "grasp_confirmed",
                     "slip_detected", "intent", "confidence", "execute_steps",
                     "escalate", "evidence"],
        "properties": {
            "phase": {"type": "string", "enum": [p.value for p in Phase]},
            "progress": {"type": "string", "enum": [p.value for p in Progress]},
            "target_visible": {"type": "boolean"},
            "grasp_confirmed": {"type": "boolean"},
            "slip_detected": {"type": "boolean"},
            "intent": {"type": "string", "enum": [i.value for i in Intent]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "execute_steps": {"type": "integer", "minimum": 0},
            "escalate": {"type": "boolean"},
            # SHORT ON PURPOSE. Generation dominates latency at 500M, and a
            # ~117-token reply put the Jetson at 3.05s against a 3.0s limit.
            # 120 characters is enough to name what was seen and be audited on
            # it; prose beyond that costs milliseconds per token and adds no
            # decision content, since every DECISION is in the typed fields.
            "evidence": {"type": "string", "maxLength": EVIDENCE_MAX_CHARS},
        },
    }


class MonitorRejected(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass
class MonitorReading:
    """A validated monitor response. Construction implies nothing is trusted yet."""
    phase: Phase
    progress: Progress
    target_visible: bool
    grasp_confirmed: bool
    slip_detected: bool
    intent: Intent
    confidence: float
    execute_steps_raw: int          # what the MODEL asked for; never used directly
    escalate_requested: bool
    evidence: str
    observed_at: float = field(default_factory=time.time)
    backend: str = "unknown"
    latency_s: float = 0.0

    @property
    def adverse(self) -> bool:
        return (self.progress in ADVERSE_PROGRESS
                or self.intent in ADVERSE_INTENT
                or self.slip_detected)

    @property
    def uncertain(self) -> bool:
        return (self.progress is Progress.UNCERTAIN
                or self.intent is Intent.UNCERTAIN
                or not self.target_visible)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("phase", "progress", "intent"):
            d[k] = getattr(self, k).value
        d.update({"schema": SCHEMA, "adverse": self.adverse,
                  "uncertain": self.uncertain,
                  "confidence_is_advisory": True})
        return d


def parse_reading(payload: Any, *, backend: str = "unknown",
                  latency_s: float = 0.0) -> MonitorReading:
    """Validate strictly. A malformed answer raises rather than degrading."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except Exception as exc:
            raise MonitorRejected("malformed_json", str(exc)[:160]) from None
    if not isinstance(payload, dict):
        raise MonitorRejected("not_an_object", f"got {type(payload).__name__}")

    schema = response_schema()
    required = schema["required"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise MonitorRejected("missing_fields", ", ".join(missing))
    extra = [k for k in payload if k not in schema["properties"]]
    if extra:
        raise MonitorRejected("unexpected_fields", ", ".join(sorted(extra)))

    def enum(cls, key):
        try:
            return cls(payload[key])
        except ValueError:
            raise MonitorRejected("bad_enum",
                                  f"{key}={payload[key]!r}") from None

    for key in ("target_visible", "grasp_confirmed", "slip_detected", "escalate"):
        if not isinstance(payload[key], bool):
            raise MonitorRejected("bad_type", f"{key} must be boolean")
    conf = payload["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
            or not 0.0 <= float(conf) <= 1.0:
        raise MonitorRejected("bad_confidence", f"{conf!r} outside [0,1]")
    steps = payload["execute_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise MonitorRejected("bad_execute_steps", f"{steps!r}")
    ev = payload["evidence"]
    if not isinstance(ev, str) or not ev.strip():
        raise MonitorRejected("empty_evidence",
                              "a decision without stated evidence is not auditable")

    return MonitorReading(
        phase=enum(Phase, "phase"), progress=enum(Progress, "progress"),
        target_visible=payload["target_visible"],
        grasp_confirmed=payload["grasp_confirmed"],
        slip_detected=payload["slip_detected"],
        intent=enum(Intent, "intent"), confidence=float(conf),
        execute_steps_raw=int(steps), escalate_requested=payload["escalate"],
        evidence=ev.strip()[:EVIDENCE_MAX_CHARS], backend=backend,
        latency_s=latency_s)


@dataclass(frozen=True)
class MonitorPolicy:
    """Deterministic limits. The model cannot widen any of these."""
    max_steps_normal: int = 8
    max_steps_uncertain: int = 3
    max_steps_contact: int = 2          # contact phases get a shorter leash
    escalate_after_adverse: int = 3     # consecutive adverse evaluations
    hold_after_uncertain: int = 4       # consecutive uncertain -> stop and look
    max_reading_age_s: float = 1.0      # older than this is stale
    hand_back_after_normal: int = 2     # consecutive good readings to hand back
    #: While Astra holds control, how often to ASK IT AGAIN. 0 = only on the
    #: transition into escalation. Calling it every adverse tick is not
    #: "reserving it for persistent escalation", it is polling an expensive
    #: reviewer for as long as things look bad -- which is the cost the local
    #: monitor exists to avoid. A re-ask should be a deliberate cadence.
    astra_recall_every: int = 0

    def ceiling(self, reading: MonitorReading) -> int:
        if reading.phase in (Phase.CONTACT, Phase.GRASP):
            return self.max_steps_contact
        if reading.uncertain:
            return self.max_steps_uncertain
        return self.max_steps_normal

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d


@dataclass
class GateDecision:
    """The DETERMINISTIC outcome. This is what the controller acts on."""
    disposition: Disposition
    execute_steps: int
    reason: str
    reading: dict[str, Any] | None = None
    proposed_steps: int | None = None
    clamped_from: int | None = None
    adverse_streak: int = 0
    uncertain_streak: int = 0
    escalate: bool = False
    shadow: bool = True
    decided_at: float = field(default_factory=time.time)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["disposition"] = self.disposition.value
        d["schema"] = SCHEMA
        d["authoritative"] = False
        d["note"] = ("advisory input to the existing state machine; Ruckig, the "
                     "RSI gateway, limits, watchdog, operator stop and the "
                     "physical E-stop remain authoritative")
        return d


class MonitorGate:
    """Holds streaks across evaluations. Shadow by default."""

    def __init__(self, policy: MonitorPolicy | None = None, *,
                 shadow: bool = True, chunk_steps: int = 50) -> None:
        self.policy = policy or MonitorPolicy()
        self.shadow = bool(shadow)
        self.chunk_steps = chunk_steps
        self.adverse_streak = 0
        self.uncertain_streak = 0
        self.normal_streak = 0
        self.escalated = False
        self.history: list[GateDecision] = []

    # -- the fail-safe path, used for every "no usable answer" ---------------
    def fail_safe(self, reason: str) -> GateDecision:
        """No answer is the most conservative answer: execute nothing.

        execute_steps=0 means the existing state machine holds -- it keeps
        answering the controller with the measured pose and STOPFLAG=1. This is
        deliberately NOT a fault: a monitor that cannot be reached is a reason to
        stop moving, not a reason to latch.
        """
        self.adverse_streak = 0
        self.normal_streak = 0
        d = GateDecision(Disposition.FAIL_SAFE, 0, reason,
                         shadow=self.shadow,
                         uncertain_streak=self.uncertain_streak)
        self.history.append(d)
        return d

    def evaluate(self, reading: MonitorReading | None, *,
                 proposed_steps: int, now: float | None = None,
                 error: str | None = None) -> GateDecision:
        now = time.time() if now is None else now
        if reading is None:
            return self.fail_safe(error or "no monitor reading available")
        age = now - reading.observed_at
        if age > self.policy.max_reading_age_s:
            return self.fail_safe(
                f"monitor reading {age:.3f}s old (limit "
                f"{self.policy.max_reading_age_s}s); a stale status describes a "
                f"scene the arm has already left")

        # streaks first: escalation is a property of history, not of one frame
        if reading.adverse:
            self.adverse_streak += 1
            self.normal_streak = 0
        else:
            self.adverse_streak = 0
        if reading.uncertain:
            self.uncertain_streak += 1
            self.normal_streak = 0
        else:
            self.uncertain_streak = 0
            if not reading.adverse:
                self.normal_streak += 1

        ceiling = self.policy.ceiling(reading)
        allowed = max(0, min(int(reading.execute_steps_raw), ceiling,
                             int(proposed_steps), self.chunk_steps))
        clamped_from = (reading.execute_steps_raw
                        if reading.execute_steps_raw != allowed else None)

        if self.adverse_streak >= self.policy.escalate_after_adverse:
            self.escalated = True
            d = GateDecision(
                Disposition.ESCALATE, 0,
                f"adverse evidence persisted {self.adverse_streak} evaluations "
                f"(limit {self.policy.escalate_after_adverse}): "
                f"progress={reading.progress.value} intent={reading.intent.value} "
                f"slip={reading.slip_detected}",
                reading.to_log(), proposed_steps, clamped_from,
                self.adverse_streak, self.uncertain_streak, True, self.shadow)
        elif self.uncertain_streak >= self.policy.hold_after_uncertain:
            d = GateDecision(
                Disposition.HOLD_REOBSERVE, 0,
                f"uncertain for {self.uncertain_streak} evaluations (limit "
                f"{self.policy.hold_after_uncertain}); stopping to re-observe "
                f"rather than acting on unclear evidence",
                reading.to_log(), proposed_steps, clamped_from,
                self.adverse_streak, self.uncertain_streak, False, self.shadow)
        elif reading.adverse or reading.uncertain:
            d = GateDecision(
                Disposition.SHORTEN, allowed,
                f"single adverse/uncertain reading -- shortening to {allowed} "
                f"step(s) rather than handing control away "
                f"(adverse {self.adverse_streak}/"
                f"{self.policy.escalate_after_adverse})",
                reading.to_log(), proposed_steps, clamped_from,
                self.adverse_streak, self.uncertain_streak, False, self.shadow)
        else:
            d = GateDecision(
                Disposition.ALLOW_PREFIX, allowed,
                f"progress normal, intent aligned; {allowed} step(s) of "
                f"{proposed_steps} permitted (ceiling {ceiling})",
                reading.to_log(), proposed_steps, clamped_from,
                self.adverse_streak, self.uncertain_streak, False, self.shadow)
        self.history.append(d)
        return d

    def may_hand_back(self) -> tuple[bool, str]:
        """After an escalation, when may pi0.5 resume?"""
        if not self.escalated:
            return True, "never escalated"
        if self.normal_streak >= self.policy.hand_back_after_normal:
            return True, (f"{self.normal_streak} consecutive normal evaluations "
                          f"(limit {self.policy.hand_back_after_normal})")
        return False, (f"recovery not verified: {self.normal_streak}/"
                       f"{self.policy.hand_back_after_normal} normal evaluations")

    def hand_back(self) -> None:
        ok, _ = self.may_hand_back()
        if ok:
            self.escalated = False

    def metrics(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for d in self.history:
            counts[d.disposition.value] = counts.get(d.disposition.value, 0) + 1
        clamped = [d for d in self.history if d.clamped_from is not None]
        lat = [d.reading["latency_s"] for d in self.history
               if d.reading and "latency_s" in d.reading]
        return {
            "schema": SCHEMA, "shadow": self.shadow,
            "evaluations": len(self.history), "dispositions": counts,
            "clamped_decisions": len(clamped),
            "clamp_examples": [{"proposed": d.clamped_from,
                                "allowed": d.execute_steps} for d in clamped[:5]],
            "adverse_streak": self.adverse_streak,
            "uncertain_streak": self.uncertain_streak,
            "normal_streak": self.normal_streak,
            "escalated": self.escalated,
            "latency_s": {"n": len(lat),
                          "mean": round(sum(lat) / len(lat), 4) if lat else None,
                          "max": round(max(lat), 4) if lat else None},
            "policy": self.policy.to_log(),
        }
