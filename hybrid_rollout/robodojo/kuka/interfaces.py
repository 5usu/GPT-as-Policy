"""The three things called "TCP" here, kept apart on purpose.

This module exists because the word is overloaded three ways in this project and
confusing any two of them leads somewhere bad. Everything below is a VERIFIED
DEPLOYMENT FACT supplied by the deployment engineer, not an inference.

    1. RSI_UDP           UDP 59152 -- the real-time trajectory control loop
    2. EXT_TRIGGER_TCP   TCP 54600 -- EthernetKRL/iicoServer, start/stop only
    3. TOOL_CENTER_POINT the $TOOL frame -- geometry, not networking

WHY IT MATTERS THAT THESE ARE SEPARATE
  - Reading "TCP 54600" as the control stream would suggest we can send a
    trajectory over it. We cannot: it starts and stops the RSI program.
  - Reading "TCP" as Tool Center Point when a port is meant, or the reverse,
    turns a networking question into a calibration question or vice versa.
  - The Tool Center Point is UNKNOWN. Nothing here may assume a value for it.

THE DECISIVE CONSTRAINT ON THIS BUILD
The deployed RSI RECEIVE configuration accepts exactly:

    AK.A1 .. AK.A6      joint corrections, degrees
    STOPFLAG            controlled stop

There is NO RKorr element. The controller therefore cannot accept a Cartesian
correction or an end-effector target in the current configuration, no matter what
we calculate, what the tool transform turns out to be, or what mode a reviewer
returns. `eef` and Astra-direct are consequently disabled by the INTERFACE, not
merely by missing calibration -- and fixing the calibration alone would not
enable them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA = "hybrid_rollout.robodojo.kuka.interfaces.v1"


class Meaning(str, Enum):
    """Which "TCP" is being talked about."""
    RSI_UDP = "rsi_udp_realtime_control"
    EXT_TRIGGER_TCP = "ethernetkrl_trigger_channel"
    TOOL_CENTER_POINT = "tool_center_point_geometry"


# --------------------------------------------------------------- 1. RSI UDP
RSI_UDP_PORT = 59152
RSI_CYCLE_S = 0.004
RSI_HZ = 250.0
RSI_BIND_SIDE = "jetson"          # the Jetson binds; the controller connects in
RSI_CONNECT_SIDE = "kuka_controller"
RSI_DESCRIPTION = (
    "Real-time trajectory control. The Jetson BINDS this UDP port and the KUKA "
    "controller connects to it, sending one frame every 4 ms. Every frame must "
    "be answered in-cycle with the same IPOC echoed back.")

#: What the deployed RSI receive configuration will actually accept.
RSI_ACCEPTED_ELEMENTS = ("AK.A1", "AK.A2", "AK.A3", "AK.A4", "AK.A5", "AK.A6",
                         "STOPFLAG")
RSI_HAS_RKORR = False
RSI_CARTESIAN_SUPPORTED = RSI_HAS_RKORR
RSI_INTERFACE_NOTE = (
    "Joint corrections and a stop flag only. No RKorr element exists, so "
    "Cartesian corrections and direct end-effector targets cannot be delivered "
    "in this configuration.")


# ------------------------------------------------------- 2. EXT trigger (TCP)
EXT_TRIGGER_PORT = 54600
EXT_TRIGGER_PROTOCOL = "tcp"
EXT_TRIGGER_STACK = "EthernetKRL / iicoServer"
EXT_TRIGGER_PURPOSE = (
    "Remote start and stop of the RSI program, used by the production --ext "
    "path. THIS IS NOT THE TRAJECTORY STREAM: no motion command travels over it "
    "and nothing in this package sends to it.")
EXT_TRIGGER_USED_BY_THIS_PACKAGE = False


# ------------------------------------------------- 3. Tool Center Point (TCP)
TOOL_TRANSFORM_KNOWN = False
TOOL_TRANSFORM_VALUE = None       # never a placeholder; unknown stays unknown
FK_FRAME = "flange"
FK_IS_TOOL_TIP = False
TOOL_NOTE = (
    "SOURCE.json records tcp_transform=UNKNOWN. The custom $TOOL has not been "
    "identified or calibrated, so forward kinematics here describe FLANGE motion "
    "and not verified tool-tip motion. Do not invent a value; it must be "
    "supplied and verified by the deployment engineer.")


# ------------------------------------------------------------- safety posture
HOLD_IS_DEFAULT = True
STOP_REPLIES_CONTINUE = True
STOP_FLAG_ON_CONTROLLED_STOP = 1
SILENCE_IS_UNSAFE = (
    "A controlled stop keeps REPLYING, with Stopflag=1. Falling silent faults "
    "the controller, so silence is a failure mode rather than a safe state.")
ESTOP_AUTHORITATIVE = (
    "The hardware E-stop is authoritative. Software may observe it and refuse; "
    "software must never clear or bypass it, and no code path here writes it.")


@dataclass(frozen=True)
class InterfaceFact:
    meaning: Meaning
    name: str
    port: int | None
    protocol: str | None
    carries_motion: bool
    description: str

    def to_log(self) -> dict[str, Any]:
        return {"meaning": self.meaning.value, "name": self.name,
                "port": self.port, "protocol": self.protocol,
                "carries_motion_commands": self.carries_motion,
                "description": self.description}


FACTS: tuple[InterfaceFact, ...] = (
    InterfaceFact(Meaning.RSI_UDP, "RSI real-time control", RSI_UDP_PORT, "udp",
                  True, RSI_DESCRIPTION),
    InterfaceFact(Meaning.EXT_TRIGGER_TCP, EXT_TRIGGER_STACK, EXT_TRIGGER_PORT,
                  "tcp", False, EXT_TRIGGER_PURPOSE),
    InterfaceFact(Meaning.TOOL_CENTER_POINT, "$TOOL / Tool Center Point", None,
                  None, False, TOOL_NOTE),
)


def describe() -> list[dict[str, Any]]:
    return [f.to_log() for f in FACTS]


def cartesian_capability() -> tuple[bool, list[str]]:
    """Can this cell accept a Cartesian correction at all? Returns (ok, reasons)."""
    blockers: list[str] = []
    if not RSI_HAS_RKORR:
        blockers.append(
            f"the deployed RSI receive configuration accepts only "
            f"{', '.join(RSI_ACCEPTED_ELEMENTS)} -- there is no RKorr element, so "
            f"a Cartesian correction cannot be delivered")
    if not TOOL_TRANSFORM_KNOWN:
        blockers.append(
            "the $TOOL/Tool Center Point transform is UNKNOWN, so a Cartesian "
            "target could not be expressed correctly even if it could be sent")
    return (not blockers), blockers


def modes_locked() -> dict[str, str]:
    """Which decision modes cannot execute here, and the reason for each."""
    ok, why = cartesian_capability()
    reason = "; ".join(why)
    return {
        "eef": reason,
        "astra_direct": (f"Astra-direct is Cartesian-only upstream "
                         f"(gpt_only_client refuses joint actions), so it "
                         f"inherits the same blocker: {reason}"),
        "edit": ("joint-space edits are deliverable over AK.A1-A6, but remain "
                 "disabled in this build pending supervised validation; only a "
                 "bounded student prefix is permitted for now"),
    }


def to_log() -> dict[str, Any]:
    ok, why = cartesian_capability()
    return {"schema": SCHEMA, "interfaces": describe(),
            "rsi_accepted_elements": list(RSI_ACCEPTED_ELEMENTS),
            "rsi_cartesian_supported": RSI_CARTESIAN_SUPPORTED,
            "tool_transform_known": TOOL_TRANSFORM_KNOWN,
            "fk_frame": FK_FRAME, "fk_is_tool_tip": FK_IS_TOOL_TIP,
            "cartesian_capable": ok, "cartesian_blockers": why,
            "modes_locked": modes_locked(),
            "hold_is_default": HOLD_IS_DEFAULT,
            "silence_is_unsafe": SILENCE_IS_UNSAFE,
            "estop": ESTOP_AUTHORITATIVE}
