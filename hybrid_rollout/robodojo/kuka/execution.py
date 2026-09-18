"""The execution state machine and the RSI protocol adapter.

    OFFLINE -> HOLD -> ARMED -> EXECUTING -> HOLD
                 ^                              |
                 +--------- FAULT (latched) <---+

MOTION IS DISABLED BY DEFAULT. Reaching ARMED needs every preflight gate to pass
AND a deliberate local operator action; neither alone is enough, and no remote
caller can arm.

SILENCE IS NEVER A STOP. In HOLD and in FAULT the adapter keeps answering every
4 ms frame -- with the measured pose and STOPFLAG=1 -- because a missing or late
reply faults the controller. Stopping means "reply, commanding nothing", not
"stop replying". Every failure path below lands there: startup, timeout, stale
IPOC, stale observation, validation failure, tolerance breach, process error,
reviewer failure.

TRANSPORT IS INJECTED. `ProtocolAdapter` speaks Rob/AIPOS/IPOC in and
Sen/AK/STOPFLAG/IPOC out over whatever transport it is handed. Tests hand it a
fake; nothing here binds the production address.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

from .cell import RSI_REPLY_TYPE
from .contract import ARM_DIM
from .interfaces import RSI_UDP_PORT, STOP_FLAG_ON_CONTROLLED_STOP
from .monitor import DeviationMonitor, Verdict
from .rsi_gateway import build_frame, parse_frame

SCHEMA = "hybrid_rollout.robodojo.kuka.execution.v1"

STOPFLAG_RUN = 0
STOPFLAG_STOP = STOP_FLAG_ON_CONTROLLED_STOP      # 1


class State(str, Enum):
    OFFLINE = "OFFLINE"        # no session; nothing bound
    HOLD = "HOLD"              # answering, commanding nothing
    ARMED = "ARMED"            # gates passed + operator armed; still not moving
    EXECUTING = "EXECUTING"    # emitting an interpolated prefix
    FAULT = "FAULT"            # latched; answers with STOPFLAG=1 until reset


#: Reasons that force HOLD. Each is a distinct observable, not a catch-all.
class HoldReason(str, Enum):
    STARTUP = "startup"
    NO_COMMAND = "no_command"
    REVIEWER_TIMEOUT = "reviewer_timeout"
    REVIEWER_FAILURE = "reviewer_failure"
    STALE_IPOC = "stale_ipoc"
    STALE_OBSERVATION = "stale_observation"
    VALIDATION_FAILED = "validation_failed"
    TOLERANCE_BREACH = "tolerance_breach"
    PROCESS_ERROR = "process_error"
    OPERATOR_DISARMED = "operator_disarmed"
    PREFLIGHT_FAILED = "preflight_failed"


class Transport(Protocol):
    """Whatever carries RSI frames. Injected; never constructed here."""

    def receive(self, timeout_s: float) -> tuple[bytes, Any] | None: ...
    def send(self, payload: bytes, peer: Any) -> None: ...


@dataclass
class FrameOutcome:
    ipoc: int | None
    state: str
    stopflag: int
    commanded: list[float] | None
    measured: list[float] | None
    reason: str
    monitor_verdict: str | None = None

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


class ProtocolAdapter:
    """Rob/AIPOS/IPOC in, Sen Type=ImFree / AK / STOPFLAG / IPOC out.

    Validates that IPOC is present and monotonic. A malformed frame is NOT
    answered with a guessed IPOC -- an unmatched reply is worse than a missed one,
    because the controller would apply it to the wrong cycle.
    """

    def __init__(self, transport: Transport, *, max_ipoc_gap: int = 50) -> None:
        self.transport = transport
        self.max_ipoc_gap = max_ipoc_gap
        self.last_ipoc: int | None = None
        self.frames_in = 0
        self.frames_out = 0
        self.malformed = 0
        self.ipoc_regressions = 0
        self.ipoc_jumps = 0

    def poll(self, timeout_s: float) -> tuple[int, list[float], Any] | None:
        got = self.transport.receive(timeout_s)
        if got is None:
            return None
        data, peer = got
        self.frames_in += 1
        ipoc, joints = parse_frame(data.decode("utf-8", "replace"))
        if ipoc is None or joints is None:
            self.malformed += 1
            return None
        if self.last_ipoc is not None:
            if ipoc <= self.last_ipoc:
                self.ipoc_regressions += 1
                return None                     # stale/replayed: do not answer it
            if ipoc - self.last_ipoc > self.max_ipoc_gap:
                self.ipoc_jumps += 1            # answered, but counted
        self.last_ipoc = ipoc
        return ipoc, joints, peer

    def reply(self, ipoc: int, joints: Sequence[float], *, stopflag: int,
              peer: Any, gripper_pos: int = 0) -> None:
        frame = build_frame(ipoc, joints, gripper_pos=gripper_pos,
                            stop_flag=stopflag)
        self.transport.send(frame.encode("ascii"), peer)
        self.frames_out += 1

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "reply_type": RSI_REPLY_TYPE,
                "port": RSI_UDP_PORT, "frames_in": self.frames_in,
                "frames_out": self.frames_out, "malformed": self.malformed,
                "ipoc_regressions": self.ipoc_regressions,
                "ipoc_jumps": self.ipoc_jumps, "last_ipoc": self.last_ipoc}


@dataclass
class ArmingRequest:
    """A deliberate LOCAL operator action. Not a config flag, not remote."""
    operator: str
    local: bool = True
    confirmed_estop_reachable: bool = False
    confirmed_area_clear: bool = False
    issued_at: float = field(default_factory=time.time)

    def valid(self) -> tuple[bool, list[str]]:
        missing = []
        if not self.operator:
            missing.append("named operator")
        if not self.local:
            missing.append("arming must be local to the cell, not remote")
        if not self.confirmed_estop_reachable:
            missing.append("operator has not confirmed the E-stop is reachable")
        if not self.confirmed_area_clear:
            missing.append("operator has not confirmed the area is clear")
        return (not missing), missing

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["valid"] = self.valid()[0]
        return d


class ExecutionController:
    """Owns the state, the adapter and the monitor. Emits nothing by default."""

    def __init__(self, adapter: ProtocolAdapter, *,
                 monitor: DeviationMonitor | None = None,
                 otg: Any = None,
                 preflight: Callable[[], tuple[bool, list[str]]] | None = None,
                 allow_motion: bool = False,
                 observation_max_age_s: float | None = None,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.adapter = adapter
        self.monitor = monitor
        self.otg = otg
        self.preflight = preflight
        self.allow_motion = bool(allow_motion)
        self.observation_max_age_s = observation_max_age_s
        self.on_event = on_event
        self.state = State.OFFLINE
        self.hold_reason: str = HoldReason.STARTUP.value
        self.fault_reason: str | None = None
        self.arming: ArmingRequest | None = None
        self._queue: list[list[float]] = []
        self._commanded_target: list[float] | None = None
        self.outcomes: list[FrameOutcome] = []

    # ------------------------------------------------------------ transitions
    def _emit(self, kind: str, **kw) -> None:
        if self.on_event:
            self.on_event({"schema": SCHEMA, "event": kind,
                           "state": self.state.value, **kw})

    def go_hold(self, reason: str | HoldReason) -> None:
        r = reason.value if isinstance(reason, HoldReason) else str(reason)
        if self.state is State.FAULT:
            return                                  # FAULT outranks HOLD
        self._queue.clear()
        self.state = State.HOLD
        self.hold_reason = r
        self._emit("hold", reason=r)

    def go_fault(self, reason: str) -> None:
        self._queue.clear()
        self.state = State.FAULT
        self.fault_reason = reason
        self._emit("fault", reason=reason)

    def open_session(self) -> None:
        """OFFLINE -> HOLD. Answering begins; nothing moves."""
        if self.state is State.FAULT:
            return
        self.state = State.HOLD
        self.hold_reason = HoldReason.STARTUP.value
        self._emit("session_open")

    def arm(self, request: ArmingRequest) -> tuple[bool, list[str]]:
        """HOLD -> ARMED. Needs preflight AND a valid local operator action."""
        blockers: list[str] = []
        if self.state is State.FAULT:
            return False, [f"latched FAULT: {self.fault_reason}"]
        if self.state is not State.HOLD:
            blockers.append(f"must be in HOLD to arm, currently {self.state.value}")
        if not self.allow_motion:
            blockers.append("controller constructed with allow_motion=False")
        ok, missing = request.valid()
        if not ok:
            blockers.extend(missing)
        if self.preflight is not None:
            pok, pblockers = self.preflight()
            if not pok:
                blockers.extend(pblockers)
        else:
            blockers.append("no preflight supplied; refusing to arm blind")
        if self.otg is None or not getattr(self.otg, "deployable", False):
            blockers.append("no deployable trajectory generator "
                            "(Ruckig with validated limits) supplied")
        if self.monitor is None:
            blockers.append("no commanded-vs-measured monitor supplied")
        if blockers:
            self.go_hold(HoldReason.PREFLIGHT_FAILED)
            self._emit("arm_refused", blockers=blockers)
            return False, blockers
        self.arming = request
        self.state = State.ARMED
        self._emit("armed", operator=request.operator)
        return True, []

    def disarm(self, reason: str = "operator") -> None:
        self.arming = None
        self.go_hold(f"{HoldReason.OPERATOR_DISARMED.value}: {reason}")

    # ------------------------------------------------------------- commanding
    def submit(self, target: Sequence[float], *,
               observation_epoch: float | None = None,
               now: float | None = None) -> tuple[bool, str]:
        """ARMED -> EXECUTING. Generates the profile and queues it."""
        now = time.time() if now is None else now
        if self.state is not State.ARMED:
            return False, f"not ARMED (currently {self.state.value})"
        if self.observation_max_age_s is not None:
            if observation_epoch is None:
                self.go_hold(HoldReason.STALE_OBSERVATION)
                return False, "no observation timestamp"
            age = now - observation_epoch
            if age > self.observation_max_age_s:
                self.go_hold(HoldReason.STALE_OBSERVATION)
                return False, f"observation {age:.3f}s old"
        measured = self.adapter and self._last_measured
        if measured is None:
            self.go_hold(HoldReason.NO_COMMAND)
            return False, "no measured pose yet; refusing to command from unknown"
        try:
            result = self.otg.generate(measured, list(target)[:ARM_DIM])
        except Exception as exc:                                  # noqa: BLE001
            self.go_hold(HoldReason.PROCESS_ERROR)
            return False, f"{type(exc).__name__}: {exc}"[:200]
        if result.error or not result.cycles:
            self.go_hold(HoldReason.VALIDATION_FAILED)
            return False, result.error or "empty profile"
        self._queue = [list(r) for r in result.cycles]
        self._commanded_target = [float(v) for v in list(target)[:ARM_DIM]]
        self.state = State.EXECUTING
        self._emit("executing", cycles=len(self._queue))
        return True, f"queued {len(self._queue)} cycles"

    _last_measured: list[float] | None = None

    # ------------------------------------------------------------- the cycle
    def serve_cycle(self, timeout_s: float = 0.05) -> FrameOutcome | None:
        """One RSI exchange. Always answers when a valid frame arrives."""
        try:
            got = self.adapter.poll(timeout_s)
        except Exception as exc:                                  # noqa: BLE001
            self.go_fault(f"transport error: {type(exc).__name__}: {exc}"[:160])
            return None
        if got is None:
            # No usable frame. Nothing to answer -- an unmatched reply would be
            # applied to the wrong cycle. Persistent silence is the controller's
            # to detect; we record it.
            if self.state is State.EXECUTING:
                self.go_hold(HoldReason.STALE_IPOC)
            return None
        ipoc, measured, peer = got
        self._last_measured = list(measured)

        stopflag = STOPFLAG_STOP
        commanded: list[float] | None = None
        verdict = None
        reason = self.fault_reason or self.hold_reason

        if self.state is State.EXECUTING and self._queue:
            commanded = self._queue.pop(0)
            if self.monitor is not None:
                rep = self.monitor.observe(
                    commanded=self._commanded_target or commanded,
                    interpolated=commanded, measured=measured, ipoc=ipoc)
                verdict = rep.verdict.value
                if rep.verdict is Verdict.FAULT:
                    self.go_fault(rep.reason)
                    commanded, stopflag, reason = None, STOPFLAG_STOP, rep.reason
                elif rep.verdict is Verdict.HOLD:
                    self.go_hold(HoldReason.TOLERANCE_BREACH)
                    commanded, stopflag, reason = None, STOPFLAG_STOP, rep.reason
                else:
                    stopflag, reason = STOPFLAG_RUN, "executing"
            else:
                stopflag, reason = STOPFLAG_RUN, "executing"
            if commanded is not None and not self._queue:
                self.state = State.ARMED
                self._emit("prefix_complete")
        elif self.state is State.EXECUTING:
            self.state = State.ARMED
            reason = "prefix complete"

        # HOLD and FAULT both answer with the MEASURED pose: stay where you are.
        payload = commanded if commanded is not None else measured
        self.adapter.reply(ipoc, payload, stopflag=stopflag, peer=peer)
        out = FrameOutcome(ipoc, self.state.value, stopflag,
                           commanded, list(measured), reason, verdict)
        self.outcomes.append(out)
        return out

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "state": self.state.value,
                "hold_reason": self.hold_reason, "fault_reason": self.fault_reason,
                "allow_motion": self.allow_motion,
                "armed_by": (self.arming.operator if self.arming else None),
                "queued_cycles": len(self._queue),
                "adapter": self.adapter.to_log(),
                "monitor": self.monitor.summary() if self.monitor else None,
                "frames_answered": len(self.outcomes),
                "stopflag_semantics": ("HOLD and FAULT answer every frame with the "
                                       "measured pose and STOPFLAG=1; silence is "
                                       "never used as a stop")}
