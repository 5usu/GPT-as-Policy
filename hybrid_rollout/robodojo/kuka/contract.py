"""Action contract for the KUKA LBR iisy 11 R1300.

Values are read off the verified assets, not assumed:

  limits      kroshu/kuka_robot_descriptions @ ec20a39e (Apache-2.0), resolved URDF
  action dims checkpoints/pi05_corrected_b8/120000 config action_feature_names
  units       DEGREES for A1..A6, gripper absolute 0..1
  space       ABSOLUTE joint targets. `use_relative_actions` is an internal
              train-time transform; postprocessed model output is absolute.

The velocity caps below are derived from the URDF joint velocity limits at the
control rate, and they matter: the unmodified pi0.5 proposal measured on real
val_ood data already exceeds joint_2's cap (6.992 deg/step vs 6.667). That is a
property of the policy, not of this code, and it is reported rather than hidden.
"""
from __future__ import annotations

import math

JOINT_NAMES: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6")
GRIPPER_NAME = "gripper"
ACTION_NAMES: tuple[str, ...] = JOINT_NAMES + (GRIPPER_NAME,)
ACTION_DIM = len(ACTION_NAMES)
ARM_DIM = len(JOINT_NAMES)

CHUNK_STEPS = 50
CONTROL_HZ = 30.0
RSI_HZ = 250.0                    # KUKA RSI cycle; the gateway interpolates up to it

ACTION_SPACE = "absolute_joint_targets_deg"
GRIPPER_RANGE = (0.0, 1.0)

ROBOT_MODEL = "KUKA LBR iisy 11 R1300"
ROBOT_MODEL_SOURCE = ("kroshu/kuka_robot_descriptions @ "
                      "ec20a39ee19caf874802c5a4d57bab17df5c0766 (Apache-2.0)")

# MECHANICAL limits from the resolved kroshu URDF, degrees.
URDF_POSITION_LIMIT_DEG: tuple[tuple[float, float], ...] = (
    (-185.0, 185.0), (-230.0, 50.0), (-150.0, 150.0),
    (-180.0, 180.0), (-110.0, 110.0), (-220.0, 220.0))

# OPERATIONAL limits actually enforced by the deployed teleoperation stack
# (KUKA/teleoperation/udp_teleoperate.py:178-179). Every joint is inset 0.5 deg
# from the mechanical limit. THESE are what this package validates against: a
# command the deployed stack would reject must not pass here, and validating
# against the wider URDF numbers would have let 0.5 deg of unreachable travel
# through on every joint.
POSITION_LIMIT_DEG: tuple[tuple[float, float], ...] = (
    (-184.5, 184.5), (-229.5, 49.5), (-149.5, 149.5),
    (-179.5, 179.5), (-109.5, 109.5), (-219.5, 219.5))
POSITION_LIMIT_SOURCE = ("KUKA/teleoperation/udp_teleoperate.py JOINT_LIMITS_MIN/MAX; "
                         "0.5 deg inset from the kroshu URDF mechanical limits")

# The robot_type string the recording/conversion stack actually uses. NOTE it is
# "iico", not "iisy": the model identification (LBR iisy 11 R1300) comes from the
# operator plus an exact match against the kroshu URDF limits, NOT from any
# string in the robot code, which never names the model.
DATASET_ROBOT_TYPE = "kuka_lbr_iico"
MODEL_IDENTIFICATION_BASIS = (
    "operator identification, corroborated by all six joint limits matching the "
    "kroshu lbr_iisy11_r1300 URDF to within the deployed 0.5 deg safety inset. "
    "The teleoperation codebase itself never names the model.")

# Gripper: the deployed stack normalises a raw integer 0..GRIPPER_SCALE to 0..1
# (udp_teleoperate.py:190). Chunk values are the NORMALISED form; the RSI frame
# carries the raw integer.
GRIPPER_SCALE = 12000.0
VELOCITY_LIMIT_RAD_S: tuple[float, ...] = (
    3.49065850398866, 3.490656, 3.49065850398866,
    4.01425727958696, 4.53785605518526, 7.50491578357562)

# Range observed in the training corpus. A SANITY ENVELOPE, NOT A SAFETY LIMIT --
# far narrower than the controller limits above and used only to notice that a
# command has left the distribution the policy was trained on.
DATA_ENVELOPE_MIN_DEG = (-93.762, -122.470, 38.261, -38.908, -35.173, -22.771)
DATA_ENVELOPE_MAX_DEG = (-45.893, -25.419, 119.444, 25.121, 55.012, 64.192)
DATA_ENVELOPE_SLACK_DEG = 5.0

# Bounded correction limits for `edit`. Upstream uses 5 cm / 0.35 rad in
# Cartesian space; we have no verified TCP, so the bound is joint-space degrees.
MAX_CORRECTION_DEG = 1.0
MAX_GRIPPER_DELTA = 0.25
MAX_CORRECTED_STEPS = 5           # upstream TAKEOVER_STEPS_MAX
MAX_STUDENT_STEPS = 15            # upstream STUDENT_STEPS_MAX

# ---------------------------------------------------------------- eef gating
# `eef` REMAINS in the schema and the parser: it is a valid upstream decision and
# dropping it would fork the decision contract. What is gated is EXECUTION. An
# eef decision is accepted, recorded, and then deterministically refused at the
# KUKA execution gate until every prerequisite below is present.
#
# All four are required. Each is a distinct missing capability, and any one of
# them absent makes a Cartesian target unsafe to turn into joint commands:
# The KUKA controller can resolve Cartesian corrections itself, so OUR having an
# IK solver is not the requirement -- knowing that the controller will do it, in
# a Cartesian-capable RSI context, is. The deployed .src and Python in
# KUKA/teleoperation are JOINT-SPACE ONLY (AIPos in, AK out, no TOOL/RIst/RKorr
# anywhere), so a Cartesian path is not demonstrated by the existing setup and
# must be attested by an operator. See eef.CartesianCapability.
TCP_TRANSFORM_VERIFIED = False   # $TOOL is on the controller; supply and verify it
RSI_CARTESIAN_CONFIGURED = False # the deployed RSI context is joint-only
WORKSPACE_CONFIGURED = False     # no verified reachable-volume bounds
COLLISION_CONFIGURED = False     # no cell/self-collision model

EEF_PREREQUISITES = {
    "tcp_transform_verified": TCP_TRANSFORM_VERIFIED,
    "rsi_cartesian_configured": RSI_CARTESIAN_CONFIGURED,
    "workspace_configured": WORKSPACE_CONFIGURED,
    "collision_configured": COLLISION_CONFIGURED,
}
EEF_EXECUTION_ENABLED = all(EEF_PREREQUISITES.values())
EEF_WITHHELD_REASON = (
    "eef accepted as an upstream decision but REFUSED at the KUKA execution "
    "gate. A Cartesian target is resolved by the CONTROLLER, so the resulting "
    "joint angles cannot be checked here before they exist; the defence is "
    "bounding the step against the measured RIst pose, which requires the cell "
    "to be attested. Requires: the $TOOL/TCP transform read off the controller "
    "and verified against RIst readback; a Cartesian-capable RSI configuration "
    "(the deployed one is joint-only); configured workspace bounds; and a "
    "collision model.")


def eef_execution_gate() -> tuple[bool, list[str], str]:
    """Deterministic. Returns (allowed, missing_prerequisites, reason)."""
    missing = sorted(k for k, ok in EEF_PREREQUISITES.items() if not ok)
    if missing:
        return False, missing, f"{EEF_WITHHELD_REASON} Missing: {', '.join(missing)}."
    return True, [], "all eef prerequisites verified"


def max_step_deg(hz: float = CONTROL_HZ) -> list[float]:
    """Largest per-step change each joint can execute at `hz`."""
    return [math.degrees(v) / hz for v in VELOCITY_LIMIT_RAD_S]
