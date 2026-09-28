"""Action contract for bimanual I2RT YAM running robocurve/pi0.5-yam.

Values are read off the published assets, not assumed:

  checkpoint   robocurve/pi0.5-yam @ ee17bb36 (model card + norm_stats.json)
  layout       [left j1..j6, left gripper, right j1..j6, right gripper]; the
               norm stats' per-dim ranges match that order (grippers at 6 and
               13 span ~[0, 1], joints match the YAM joint ranges)
  units        RADIANS for the twelve arm joints, gripper normalised 0..1
  gripper      0 = CLOSED, 1 = OPEN. The OPPOSITE of the KUKA cell, where the
               monitor's event detector treats >= 0.5 as closing.
  space        ABSOLUTE joint targets; (16, 14) chunks at 30 Hz
  cameras      top, left, right -- in that order -- 360x640 source frames
  limits       i2rt-robotics/i2rt @ 120c3c81 arm/yam/v1/yam.xml joint ranges

WHAT CHANGED FROM THE KUKA CONTRACT, AND WHY
  two arms instead of one, so every per-joint rule runs over JOINT_INDICES and
  every gripper rule over GRIPPER_INDICES; radians instead of degrees; a
  16-step chunk instead of 50; and the gripper polarity flips.

GRIPPER STATE IS THE LAST COMMAND, NOT THE ENCODER.
The MolmoAct2 YAM recordings logged the commanded opening in the state's
gripper channels (vla-edge PolicySpec `gripper_state="commanded"` for this
checkpoint). A gripper closed on a block reads a small positive opening on the
encoder; feeding that to the policy shows it a state it never trained on.
"""
from __future__ import annotations

import math

ARMS: tuple[str, ...] = ("left", "right")
JOINTS_PER_ARM = 6
ACTION_NAMES: tuple[str, ...] = tuple(
    name for arm in ARMS for name in
    [f"{arm}_joint{i + 1}" for i in range(JOINTS_PER_ARM)] + [f"{arm}_gripper"])
ACTION_DIM = len(ACTION_NAMES)                     # 14

#: Row indices. Every rule that was "range(ARM_DIM)" on the KUKA is one of these.
ARM_JOINT_INDICES: dict[str, tuple[int, ...]] = {
    "left": tuple(range(0, 6)), "right": tuple(range(7, 13))}
GRIPPER_INDICES: dict[str, int] = {"left": 6, "right": 13}
JOINT_INDICES: tuple[int, ...] = ARM_JOINT_INDICES["left"] + ARM_JOINT_INDICES["right"]

CHUNK_STEPS = 16
CONTROL_HZ = 30.0

ACTION_SPACE = "absolute_joint_targets_rad"
GRIPPER_RANGE = (0.0, 1.0)
GRIPPER_CONVENTION = "closed_0_open_1"
GRIPPER_STATE_SOURCE = "commanded"
#: Midpoint used to name a gripper transition. Below it the gripper is closing.
GRIPPER_THRESHOLD = 0.5

#: Training order. Order is load-bearing and silent when wrong: openpi maps
#: top -> base_0_rgb, left -> left_wrist_0_rgb, right -> right_wrist_0_rgb.
CAMERA_NAMES: tuple[str, ...] = ("top", "left", "right")
#: Source frame size the checkpoint was trained on (H, W). Frames are resized
#: with padding to 224x224, so a 4:3 capture pads differently from a 16:9 one
#: and shows the policy a letterbox it never saw.
IMAGE_SOURCE_HW = (360, 640)

ROBOT_MODEL = "I2RT YAM bimanual (2x 6-DoF + linear gripper)"
ROBOT_MODEL_SOURCE = ("i2rt-robotics/i2rt @ 120c3c81400171174604e503943f8d1ebc891058 "
                      "i2rt/robot_models/arm/yam/v1 (MIT)")

# Joint ranges from yam.xml, radians, per arm joint 1..6.
MODEL_POSITION_RANGE_RAD: tuple[tuple[float, float], ...] = (
    (-2.61799, 3.14159), (0.0, 3.66519), (0.0, 3.14159),
    (-1.69297, 1.5708), (-1.5708, 1.5708), (-2.0944, 2.0944))
#: get_yam_robot widens the model range by 0.15 rad and CLIPS commands to the
#: result. These are what the deployed driver enforces, so they are what this
#: package validates against -- the same choice the KUKA branch made with the
#: teleop stack's 0.5 deg inset. The model range itself is too tight to use:
#: the training corpus sits at joint2 = -0.0006 (q01), below its 0.0 bound.
#: Anything outside this is REFUSED here rather than clipped by the driver,
#: because a clipped joint command is a different trajectory.
DRIVER_BUFFER_RAD = 0.15
ARM_POSITION_LIMIT_RAD: tuple[tuple[float, float], ...] = tuple(
    (lo - DRIVER_BUFFER_RAD, hi + DRIVER_BUFFER_RAD)
    for lo, hi in MODEL_POSITION_RANGE_RAD)
POSITION_LIMIT_SOURCE = ("i2rt get_yam_robot: yam.xml joint ranges widened by "
                         "0.15 rad, the range the driver clips commands to")

#: Per-joint speed ceiling. The published YAM pi0.5 controller in
#: Agents2AgentsAI/vla-edge (examples/bimanual-yam/pi05_motion.py MotionLimits)
#: runs at 2.2 rad/s with 6 rad/s^2 and 60 rad/s^3. It is a reference
#: deployment's setting, not a measured property of THIS rig.
MAX_JOINT_VELOCITY_RAD_S = 2.2
MAX_JOINT_VELOCITY_SOURCE = "vla-edge pi05_motion.MotionLimits default"

#: The training corpus range, straight from the checkpoint's norm stats
#: (actions q01/q99). A SANITY ENVELOPE, NOT A SAFETY LIMIT.
DATA_ENVELOPE_Q01: tuple[float, ...] = (
    -0.7585, 0.0004, 0.0021, -1.688, -0.4479, -1.6473, 0.0,
    -0.5497, -0.0006, 0.0036, -1.6358, -1.4599, -0.617, 0.0095)
DATA_ENVELOPE_Q99: tuple[float, ...] = (
    0.4497, 2.5051, 2.7945, 0.9932, 1.5787, 0.5434, 0.9998,
    0.9361, 2.5076, 2.4618, 0.9405, 0.4266, 1.8557, 0.9998)
DATA_ENVELOPE_SLACK_RAD = 0.1

# Bounded correction limits for `edit`. Same magnitudes as the KUKA branch
# (1 deg per joint), expressed in radians. Upstream's edit is Cartesian; there
# is no calibrated TCP or arm-to-world frame on this rig, so it is joint space.
MAX_CORRECTION_RAD = math.radians(1.0)
MAX_GRIPPER_DELTA = 0.25
MAX_CORRECTED_STEPS = 5           # upstream TAKEOVER_STEPS_MAX
MAX_STUDENT_STEPS = 15            # upstream STUDENT_STEPS_MAX

#: The checkpoint this contract describes. The pi0.5 server reports these in
#: every response and the client refuses a response that does not match.
CHECKPOINT = {
    "repo_id": "robocurve/pi0.5-yam",
    "revision": "ee17bb361e95eeba57853a2840480f5a1fc81a84",
    "framework": "openpi (JAX/Orbax)",
    "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "norm_asset_id": "yam-bimanual-merged",
    "norm_stats_sha256": "16daf28cec63d4829f01d7858bfed079ad18e183ce826a268f66c6669f323863",
    "action_horizon": CHUNK_STEPS,
    "action_dim": ACTION_DIM,
    "action_space": ACTION_SPACE,
    "gripper_convention": GRIPPER_CONVENTION,
    "cameras": list(CAMERA_NAMES),
    "license": "gemma",
    "validation": ("open-loop MSE on one held-out session only; no closed-loop "
                   "or real-robot success rate published"),
}

# ---------------------------------------------------------------- eef gating
# `eef` stays in the schema and the parser for the same reason as on the KUKA:
# dropping it would fork the upstream decision contract. Execution is refused
# until every prerequisite below exists. None does on this rig.
TCP_TRANSFORM_VERIFIED = False    # gripper frame in yam.urdf is a mount, not a tip
ARM_BASE_FRAMES_CALIBRATED = False  # left/right base poses in a shared frame
IK_VERIFIED = False               # no IK is shipped or validated here
WORKSPACE_CONFIGURED = False
COLLISION_CONFIGURED = False      # two arms can collide with each other

EEF_PREREQUISITES = {
    "tcp_transform_verified": TCP_TRANSFORM_VERIFIED,
    "arm_base_frames_calibrated": ARM_BASE_FRAMES_CALIBRATED,
    "ik_verified": IK_VERIFIED,
    "workspace_configured": WORKSPACE_CONFIGURED,
    "collision_configured": COLLISION_CONFIGURED,
}
EEF_EXECUTION_ENABLED = all(EEF_PREREQUISITES.values())
EEF_WITHHELD_REASON = (
    "eef accepted as an upstream decision but REFUSED at the YAM execution "
    "gate. A Cartesian target needs a verified tool-tip transform, both arm "
    "base frames in one calibrated frame, a validated IK, workspace bounds and "
    "an inter-arm collision model. None exists on this rig.")


def eef_execution_gate() -> tuple[bool, list[str], str]:
    """Deterministic. Returns (allowed, missing_prerequisites, reason)."""
    missing = sorted(k for k, ok in EEF_PREREQUISITES.items() if not ok)
    if missing:
        return False, missing, f"{EEF_WITHHELD_REASON} Missing: {', '.join(missing)}."
    return True, [], "all eef prerequisites verified"


def max_step_rad(hz: float = CONTROL_HZ) -> float:
    """Largest per-step change any arm joint may make at `hz`."""
    return MAX_JOINT_VELOCITY_RAD_S / hz


def arm_joint_limit(index: int) -> tuple[float, float]:
    """Position limit for a row index in JOINT_INDICES."""
    for joints in ARM_JOINT_INDICES.values():
        if index in joints:
            return ARM_POSITION_LIMIT_RAD[joints.index(index)]
    raise IndexError(f"row index {index} is not an arm joint")


def split_arms(row) -> dict[str, dict[str, object]]:
    """{'left': {'joints': [6], 'gripper': g}, 'right': {...}} from one row."""
    r = list(row)
    return {arm: {"joints": [r[i] for i in ARM_JOINT_INDICES[arm]],
                  "gripper": r[GRIPPER_INDICES[arm]]} for arm in ARMS}
