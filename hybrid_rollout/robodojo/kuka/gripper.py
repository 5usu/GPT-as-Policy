"""The Modbus gripper: a separate actuator on a separate bus, separately gated.

VERIFIED CELL FACT: the gripper is driven by the Jetson over /dev/ttyUSB0 -- a
CH340 USB-serial bridge to Modbus RTU on RS485 -- and does NOT pass through the
KUKA controller. The <GRIPPER_POS> element inside an RSI frame is inert here.

THE CONSEQUENCE THAT DRIVES THIS ENTIRE MODULE
An arm STOPFLAG stops the ARM. The gripper never sees it: different bus,
different protocol, no shared interlock. So after a controlled stop, or a FAULT,
or an E-stop, the gripper holds whatever it was last told. If it was closing on
the door handle, it stays closed on the door handle while the arm is frozen.

That is not a stop. It is half a stop, and the half that is still gripping is
attached to the thing you were trying to get away from.

So a stop must decide EXPLICITLY what the gripper does, and that decision needs a
direction -- which requires knowing polarity. Until polarity is verified, the
safe action is unknowable, and an unknowable safe action means NO gripper command
at all, including on stop. Refusing to move a gripper is always survivable;
moving it the wrong way is not.

Nothing here opens a serial port.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .cell import GRIPPER_BUS, GRIPPER_DEVICE, GRIPPER_PATH, GRIPPER_VIA_CONTROLLER

SCHEMA = "hybrid_rollout.robodojo.kuka.gripper.v2"


class GripperState(str, Enum):
    UNKNOWN = "unknown"        # never observed; the honest default
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"
    HOLDING = "holding"        # last command still applied, arm may be stopped


class SafeAction(str, Enum):
    HOLD = "hold"              # leave it exactly as it is
    OPEN = "open"              # release
    UNKNOWN = "unknown"        # not configured -> refuse everything


class GripperRefused(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass
class GripperCapability:
    """Everything an operator must verify before this device may be commanded."""
    device_id: Any = None                  # verified identity of /dev/ttyUSB0
    polarity: Any = None                   # which value is OPEN, which CLOSED
    open_close_limits: Any = None
    speed_force_limits: Any = None
    safe_action: SafeAction = SafeAction.UNKNOWN
    observed_state_ack: bool = False       # a real readback was seen
    enabled: bool = False                  # explicit opt-in
    device: str = GRIPPER_DEVICE
    bus: str = GRIPPER_BUS
    path: str = GRIPPER_PATH

    def missing(self) -> list[str]:
        out = []
        if self.device_id in (None, "", [], {}):
            out.append("gripper_device_id -- USB enumeration order is not stable, "
                       "so /dev/ttyUSB0 must be identity-verified, not assumed")
        if self.polarity in (None, "", [], {}):
            out.append("gripper_polarity -- which commanded value is OPEN and which "
                       "is CLOSED. Reversed means gripping when you meant to "
                       "release, on a door handle")
        if self.open_close_limits in (None, "", [], {}):
            out.append("gripper_open_close_limits")
        if self.speed_force_limits in (None, "", [], {}):
            out.append("gripper_speed_force_limits")
        if self.safe_action is SafeAction.UNKNOWN:
            out.append("gripper_safe_action -- what it must do on arm HOLD/FAULT/"
                       "E-stop. An arm stop does not reach this bus")
        if not self.observed_state_ack:
            out.append("observed_state_ack -- a real state readback has never been "
                       "seen, so commanded state cannot be confirmed")
        if not self.enabled:
            out.append("enabled=False (explicit opt-in required)")
        return out

    def allowed(self) -> tuple[bool, list[str]]:
        m = self.missing()
        return (not m), m

    def direction_verified(self) -> bool:
        """Can we state which way is 'release'? Everything on stop depends on it."""
        return self.polarity not in (None, "", [], {}) and self.observed_state_ack

    def to_log(self) -> dict[str, Any]:
        ok, missing = self.allowed()
        d = asdict(self)
        d["safe_action"] = self.safe_action.value
        d.update({"schema": SCHEMA, "allowed": ok, "missing": missing,
                  "direction_verified": self.direction_verified(),
                  "via_controller": GRIPPER_VIA_CONTROLLER,
                  "sees_arm_stopflag": False})
        return d


@dataclass
class GripperCommand:
    intent: str
    normalised: float | None
    raw: int | None = None
    issued_at: float = field(default_factory=time.time)
    emitted: bool = False
    refused_code: str | None = None
    refused_reason: str | None = None
    trigger: str = "policy"          # policy | arm_hold | arm_fault | estop

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"schema": SCHEMA, "actuator": "gripper", "bus": GRIPPER_BUS,
                  "separate_from_arm": True})
        return d


class GripperController:
    """Gates and audits every gripper command. Emits nothing without a transport."""

    def __init__(self, capability: GripperCapability, *, transport=None) -> None:
        self.cap = capability
        self.transport = transport
        self.state = GripperState.UNKNOWN
        self.last_command: GripperCommand | None = None
        self.log: list[GripperCommand] = []

    def _record(self, cmd: GripperCommand) -> GripperCommand:
        self.log.append(cmd)
        if cmd.emitted:
            self.last_command = cmd
            self.state = GripperState.HOLDING
        return cmd

    def command(self, normalised: float | None, *, trigger: str = "policy"
                ) -> GripperCommand:
        """Plan, gate and audit one command. FAIL-CLOSED at every step."""
        if normalised is None:
            return self._record(GripperCommand(
                "hold", None, trigger=trigger, refused_code="no_value",
                refused_reason="no gripper value supplied"))
        v = float(normalised)
        intent = "close" if v >= 0.5 else "open"
        cmd = GripperCommand(intent, v, trigger=trigger)
        ok, missing = self.cap.allowed()
        if not ok:
            cmd.refused_code = "not_attested"
            cmd.refused_reason = "; ".join(missing)
            return self._record(cmd)
        if self.transport is None:
            cmd.refused_code = "no_transport"
            cmd.refused_reason = ("no gripper transport wired in this build; "
                                  "planned and audited, not sent")
            return self._record(cmd)
        lim = self.cap.open_close_limits or {}
        raw = int(v * float(lim.get("closed", 0) or 0)) if lim else None
        cmd.raw, cmd.emitted = raw, True
        self.transport(raw)
        return self._record(cmd)

    def on_arm_stop(self, event: str) -> GripperCommand:
        """Arm entered HOLD/FAULT, or an E-stop was observed.

        The arm has stopped. This bus did not see that, so the gripper is still
        doing whatever it was doing. What happens now is a configured decision,
        and it requires a verified direction -- without one we cannot say which
        way 'release' is, so we command nothing and say so loudly.
        """
        if not self.cap.direction_verified():
            return self._record(GripperCommand(
                "hold", None, trigger=event, refused_code="direction_unverified",
                refused_reason=(
                    f"arm {event}: gripper polarity is not verified, so the safe "
                    f"direction is unknown. Commanding nothing. THE GRIPPER IS "
                    f"STILL IN ITS LAST COMMANDED STATE "
                    f"({self.state.value}) and this bus never saw the arm stop.")))
        if self.cap.safe_action is SafeAction.UNKNOWN:
            return self._record(GripperCommand(
                "hold", None, trigger=event, refused_code="safe_action_unknown",
                refused_reason=(f"arm {event}: no configured gripper_safe_action. "
                                f"Commanding nothing; gripper remains "
                                f"{self.state.value}.")))
        if self.cap.safe_action is SafeAction.HOLD:
            return self._record(GripperCommand(
                "hold", None, trigger=event, refused_code=None,
                refused_reason=(f"arm {event}: configured safe action is HOLD; "
                                f"gripper deliberately left as-is")))
        return self.command(0.0, trigger=event)      # configured OPEN = release

    def status(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "state": self.state.value,
                "capability": self.cap.to_log(),
                "commands_logged": len(self.log),
                "commands_emitted": sum(1 for c in self.log if c.emitted),
                "last": self.last_command.to_log() if self.last_command else None}


def stop_semantics() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "rsi_stopflag_affects_arm": True,
        "rsi_stopflag_affects_gripper": False,
        "why": ("the gripper is on a separate Modbus RTU bus and never sees the "
                "RSI STOPFLAG"),
        "consequence": ("after an arm stop the gripper holds its last commanded "
                        "state. If it was closing on the handle, it stays closed "
                        "on the handle while the arm is frozen"),
        "requirement": ("every stop procedure must decide explicitly what the "
                        "gripper does, and that decision needs a VERIFIED "
                        "polarity or it cannot be expressed as a direction"),
    }
