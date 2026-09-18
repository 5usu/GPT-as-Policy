"""The gripper as a SEPARATE actuator, gated and audited on its own.

VERIFIED CELL FACT: in this cell the gripper is driven by the Jetson over
/dev/ttyUSB0 -- a CH340 USB serial bridge to Modbus RTU on RS485. It DOES NOT
pass through the KUKA controller. The <GRIPPER_POS> element inside an RSI frame
is inert here.

WHY THAT DESERVES ITS OWN MODULE RATHER THAN A FIELD IN THE ACTION ROW
The arm and the gripper are two independent devices on two independent buses
with two independent failure modes, and treating the gripper as "column seven"
hides that. Specifically:

  - a controlled stop on the RSI loop stops the ARM. It does nothing whatsoever
    to a gripper on a separate serial bus, which will hold whatever it was last
    told. Closing on something and then E-stopping leaves it closed.
  - RSI faults if the Jetson is late; Modbus does not, so a wedged gripper write
    produces no controller-side symptom at all.
  - the arm's limits are joint angles; the gripper's are open/closed states and
    a polarity that is NOT yet known for this cell.

So gripper commands are gated separately, audited separately, and refused by
default. Nothing here opens a serial port.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .cell import GRIPPER_BUS, GRIPPER_DEVICE, GRIPPER_PATH, GRIPPER_VIA_CONTROLLER

SCHEMA = "hybrid_rollout.robodojo.kuka.gripper.v1"


class GripperRefused(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass
class GripperCapability:
    """What the operator has attested. Polarity is the dangerous unknown."""
    polarity: Any = None            # which commanded value is OPEN, which CLOSED
    device: str = GRIPPER_DEVICE
    bus: str = GRIPPER_BUS
    path: str = GRIPPER_PATH
    max_force: Any = None
    enabled: bool = False           # explicit opt-in, never default

    def missing(self) -> list[str]:
        out = []
        if self.polarity in (None, "", {}, []):
            out.append("gripper_polarity -- which commanded value is OPEN and "
                       "which is CLOSED. Getting this backwards means gripping "
                       "when you meant to release, on a door handle.")
        if self.max_force in (None, "", {}, []):
            out.append("max_force / grip limit")
        if not self.enabled:
            out.append("enabled=False (explicit opt-in required)")
        return out

    def allowed(self) -> tuple[bool, list[str]]:
        m = self.missing()
        return (not m), m

    def to_log(self) -> dict[str, Any]:
        ok, missing = self.allowed()
        d = asdict(d) if False else asdict(self)
        d["allowed"] = ok
        d["missing"] = missing
        d["via_controller"] = GRIPPER_VIA_CONTROLLER
        d["independent_of_rsi_stop"] = True
        d["note"] = ("a controlled stop on the RSI loop stops the ARM only; a "
                     "gripper on a separate bus holds its last commanded state")
        return d


@dataclass
class GripperCommand:
    """One gripper action, audited whether or not it is ever emitted."""
    intent: str                     # open | close | hold
    normalised: float | None        # 0..1 as it appears in an action row
    raw: int | None = None          # device units, once polarity is known
    issued_at: float = field(default_factory=time.time)
    emitted: bool = False
    refused_reason: str | None = None

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["actuator"] = "gripper"
        d["bus"] = GRIPPER_BUS
        d["separate_from_arm"] = True
        return d


def plan(normalised: float, capability: GripperCapability) -> GripperCommand:
    """Turn a 0..1 action-row value into an audited command. FAIL-CLOSED."""
    if normalised is None:
        return GripperCommand("hold", None, refused_reason="no gripper value")
    v = float(normalised)
    intent = "hold"
    if v >= 0.5:
        intent = "close"
    elif v < 0.5:
        intent = "open"
    cmd = GripperCommand(intent, v)
    ok, missing = capability.allowed()
    if not ok:
        cmd.refused_reason = ("gripper not attested: " + "; ".join(missing))
        return cmd
    # Polarity is attested, so the mapping can be made explicit rather than
    # assumed. Still not emitted here -- emission needs a transport.
    cmd.raw = None
    cmd.refused_reason = ("no gripper transport wired in this build; command "
                          "planned and audited, not sent")
    return cmd


def stop_semantics() -> dict[str, Any]:
    """What a controlled stop does, and does not do, to the gripper."""
    return {
        "schema": SCHEMA,
        "rsi_stop_affects_arm": True,
        "rsi_stop_affects_gripper": False,
        "why": ("the gripper is on a separate Modbus RTU bus and does not see "
                "the RSI STOPFLAG at all"),
        "consequence": ("after a controlled stop the gripper holds its last "
                        "commanded state. If it was closing on the handle, it "
                        "stays closed. Releasing is a separate deliberate act."),
        "requirement": ("any stop procedure must decide explicitly what the "
                        "gripper should do, and that decision must be recorded"),
    }
