"""Verified live-cell facts for the KUKA dishwasher cell.

Everything here was reported from the Jetson by the deployment engineer. These
are FACTS ABOUT THE CELL, not permission to activate anything. Recording that a
path exists is not the same as opening it, and nothing in this module performs
I/O -- `verify()` only inspects locally observable state.

AUTHORITY BOUNDARY, which every other design decision follows from:

    A800    produces proposals. NO route to the robot network. Cannot command.
    Jetson  validates, interpolates, gates, logs, and ALONE may execute.

The Jetson's RSI socket is currently unbound and the controller-side RSI program
is not running (its trigger port is closed), so at this moment no process
anywhere can command the arm. That is the correct resting state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA = "hybrid_rollout.robodojo.kuka.cell.v1"

# ------------------------------------------------------------------ networking
ROBOT_NIC = "eno1"
JETSON_ROBOT_IP = "172.17.255.2"
JETSON_ROBOT_CIDR = "172.17.255.2/16"
KUKA_KLI_IP = "172.17.255.1"
MEASURED_RTT_MS = 0.79
MEASURED_PACKET_LOSS = 0.0
ISOLATED_NICS = ("wlp1s0", "outside-world NIC")
NETWORK_NOTE = (
    f"{ROBOT_NIC} is a dedicated robot NIC at {JETSON_ROBOT_CIDR}; the KUKA KLI "
    f"is {KUKA_KLI_IP}, measured at ~{MEASURED_RTT_MS} ms with no loss. Wi-Fi and "
    f"the outside-world NIC are deliberately isolated from the robot path, which "
    f"is why an inference container cannot reach the arm even if it wanted to.")

# ------------------------------------------------------------------ RSI stream
#: The controller is the UDP CLIENT. It sends <Rob> carrying AIPOS + IPOC; the
#: Jetson must answer every 4 ms with <Sen Type="ImFree"> carrying AK.A1..A6,
#: STOPFLAG and the IDENTICAL IPOC. The echo is what binds a reply to its cycle.
RSI_CLIENT = "kuka_controller"
RSI_SERVER = "jetson"
RSI_REQUEST_ELEMENTS = ("AIPOS", "IPOC")
RSI_REPLY_ELEMENTS = ("AK.A1", "AK.A2", "AK.A3", "AK.A4", "AK.A5", "AK.A6",
                      "STOPFLAG", "IPOC")
RSI_REPLY_TYPE = "ImFree"
RSI_LATE_REPLY_FAULTS_CONTROLLER = True

# ------------------------------------------------------------------ rate stack
POLICY_HZ_PI05 = (1.0, 3.0)      # pi0.5 inference, measured range
POLICY_HZ_ACT = 30.0
RSI_HZ = 250.0
RSI_DT = 0.004
INTERPOLATOR = "Ruckig"
INTERPOLATOR_ARGS = ("NUM_JOINTS", RSI_DT)
INTERPOLATION_NOTE = (
    f"pi0.5 emits at {POLICY_HZ_PI05[0]}-{POLICY_HZ_PI05[1]} Hz and ACT at "
    f"{POLICY_HZ_ACT:.0f} Hz, while RSI demands {RSI_HZ:.0f} Hz. The Jetson fills "
    f"the gap with Ruckig(NUM_JOINTS, {RSI_DT}). The interpolated trajectory must "
    f"be validated against measured limits AT EVERY CYCLE, not once per chunk: "
    f"a chunk that is feasible on average can still contain an infeasible cycle.")

# --------------------------------------------------------------------- gripper
GRIPPER_DEVICE = "/dev/ttyUSB0"
GRIPPER_BRIDGE = "CH340 USB serial"
GRIPPER_BUS = "Modbus RTU over RS485"
GRIPPER_PATH = "direct_modbus"          # --gripper direct
GRIPPER_VIA_CONTROLLER = False
GRIPPER_NOTE = (
    f"The gripper is driven by the Jetson over {GRIPPER_DEVICE} "
    f"({GRIPPER_BRIDGE} -> {GRIPPER_BUS}). IT DOES NOT PASS THROUGH THE KUKA "
    f"CONTROLLER in this cell. A GRIPPER_POS element inside an RSI frame is "
    f"therefore inert here, and gripper commands must be gated and audited as a "
    f"SEPARATE actuator with its own failure modes.")

# --------------------------------------------------------------------- cameras
CAMERA_VENDOR = "Tera USB"
CAMERA_PRESENT_NODES = ("/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3")
CAMERA_CAPTURE_NODES = ("/dev/video0", "/dev/video2")
CAMERA_NODE_PAIRS = {"/dev/video0": "/dev/video1", "/dev/video2": "/dev/video3"}
CAMERA_NOTE = (
    "Two physical Tera cameras expose FOUR video nodes: 0/1 and 2/3. Only the "
    "first of each pair captures; the second is a metadata node. Which physical "
    "camera is base and which is wrist is NOT discoverable from the node number, "
    "so an explicit logical mapping is required and guessing is refused.")

# ----------------------------------------------------------------------- ports
PORT_RSI_UDP = 59152
PORT_EXT_TRIGGER_TCP = 54600
PORT_OPCUA_TCP = 4840
PORT_FACTS = {
    PORT_RSI_UDP: ("udp", "RSI real-time control", True),
    PORT_EXT_TRIGGER_TCP: ("tcp", "EthernetKRL iicoServer program trigger (--ext)", False),
    PORT_OPCUA_TCP: ("tcp", "OPC UA supervision ONLY -- never robot control", False),
}
EXT_TRIGGER_OPEN = False          # observed closed
RSI_PROGRAM_RUNNING = False       # implied by the trigger port being closed
JETSON_RSI_SOCKET_BOUND = False   # observed unbound

# ------------------------------------------------------------------- authority
AUTHORITY = {
    "a800": {"may_command": False, "role": "proposals only",
             "route_to_robot_network": False},
    "eng1": {"may_command": False, "role": "development and offline analysis",
             "route_to_robot_network": False},
    "jetson": {"may_command": True,
               "role": "validate, interpolate, gate, log, execute",
               "route_to_robot_network": True},
}

# -------------------------------------------------------- what is still absent
KNOWN_ABSENT = (
    "the CLI-to-RSI output path is not wired: the Astra loop reads recorded "
    "frames and writes audit output, and the live camera layer is only the INPUT "
    "side",
    "the 16 measured deployment values are not supplied",
    "the controller-side RSI program is not running",
    "the Jetson RSI socket is unbound",
)


@dataclass
class CellCheck:
    name: str
    passed: bool | None      # None = cannot be determined from here
    detail: str

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


def verify() -> list[CellCheck]:
    """Local, read-only assertions about the recorded facts. Opens nothing."""
    out: list[CellCheck] = []
    out.append(CellCheck(
        "robot_network_isolated", True,
        f"{ROBOT_NIC} {JETSON_ROBOT_CIDR} <-> KLI {KUKA_KLI_IP}; "
        f"{', '.join(ISOLATED_NICS)} isolated from it"))
    out.append(CellCheck(
        "rsi_direction", True,
        f"{RSI_CLIENT} is the UDP client; {RSI_SERVER} answers every {RSI_DT*1000:.0f} ms "
        f"with Type={RSI_REPLY_TYPE} and the identical IPOC"))
    out.append(CellCheck(
        "ext_trigger_closed", not EXT_TRIGGER_OPEN,
        f"tcp/{PORT_EXT_TRIGGER_TCP} closed -> the controller-side RSI program is "
        f"not running"))
    out.append(CellCheck(
        "opcua_not_control", True,
        f"tcp/{PORT_OPCUA_TCP} is OPC UA supervision only and must never be "
        f"treated as robot control"))
    out.append(CellCheck(
        "rsi_socket_unbound", not JETSON_RSI_SOCKET_BOUND,
        "the Jetson RSI socket is unbound, so no process can command the arm now"))
    out.append(CellCheck(
        "gripper_is_separate_actuator", not GRIPPER_VIA_CONTROLLER,
        GRIPPER_NOTE.split(". ")[0]))
    out.append(CellCheck(
        "authority_boundary", True,
        "only the Jetson may command; the A800 proposes and has no route"))
    out.append(CellCheck(
        "interpolation_declared", True,
        f"{INTERPOLATOR}{INTERPOLATOR_ARGS} fills {RSI_HZ:.0f} Hz from "
        f"{POLICY_HZ_PI05[0]}-{POLICY_HZ_PI05[1]} Hz"))
    for item in KNOWN_ABSENT:
        out.append(CellCheck("absent", False, item))
    return out


def to_log() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "network": {"nic": ROBOT_NIC, "jetson": JETSON_ROBOT_CIDR,
                    "kuka_kli": KUKA_KLI_IP, "rtt_ms": MEASURED_RTT_MS,
                    "packet_loss": MEASURED_PACKET_LOSS,
                    "isolated": list(ISOLATED_NICS)},
        "rsi": {"client": RSI_CLIENT, "server": RSI_SERVER,
                "request_elements": list(RSI_REQUEST_ELEMENTS),
                "reply_elements": list(RSI_REPLY_ELEMENTS),
                "reply_type": RSI_REPLY_TYPE, "hz": RSI_HZ, "dt": RSI_DT,
                "late_reply_faults_controller": RSI_LATE_REPLY_FAULTS_CONTROLLER},
        "rates": {"pi05_hz": list(POLICY_HZ_PI05), "act_hz": POLICY_HZ_ACT,
                  "rsi_hz": RSI_HZ, "interpolator": INTERPOLATOR,
                  "note": INTERPOLATION_NOTE},
        "gripper": {"device": GRIPPER_DEVICE, "bus": GRIPPER_BUS,
                    "path": GRIPPER_PATH, "via_controller": GRIPPER_VIA_CONTROLLER,
                    "note": GRIPPER_NOTE},
        "cameras": {"vendor": CAMERA_VENDOR,
                    "capture_nodes": list(CAMERA_CAPTURE_NODES),
                    "node_pairs": CAMERA_NODE_PAIRS, "note": CAMERA_NOTE},
        "ports": {str(p): {"protocol": v[0], "purpose": v[1], "robot_control": v[2]}
                  for p, v in PORT_FACTS.items()},
        "state": {"ext_trigger_open": EXT_TRIGGER_OPEN,
                  "rsi_program_running": RSI_PROGRAM_RUNNING,
                  "jetson_rsi_socket_bound": JETSON_RSI_SOCKET_BOUND},
        "authority": AUTHORITY,
        "known_absent": list(KNOWN_ABSENT),
        "checks": [c.to_log() for c in verify()],
    }
