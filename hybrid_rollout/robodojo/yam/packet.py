"""The Astra review packet for YAM, on the UNCHANGED upstream gate.

Mirrors `kuka.packet.build_packet`: upstream GATE_INSTRUCTION verbatim, then a
contract note for this robot, the provenance statement, the proposal, the FK
preview and the frame references. Returns a dict; sends nothing.

Upstream was already dual-arm, so the contract note here is closer to upstream
than the KUKA one: left/right are real, and only the joint-space edit and the
eef refusal differ.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..kuka.packet import PROVENANCE_NOTE as _KUKA_PROVENANCE
from ..robodojo_server.gate_assessment import GATE_INSTRUCTION  # upstream, verbatim
from .contract import (ACTION_NAMES, ARMS, CAMERA_NAMES, CHECKPOINT, CONTROL_HZ,
                       GRIPPER_INDICES, GRIPPER_RANGE, MAX_CORRECTED_STEPS,
                       MAX_CORRECTION_RAD, MAX_STUDENT_STEPS, ROBOT_MODEL,
                       split_arms)
from .schema import response_schema

SCHEMA = "hybrid_rollout.robodojo.yam.packet.v1"
PREVIEW_STEPS = 10

YAM_CONTRACT_NOTE = f"""ROBOT AND ACTION CONTRACT
{ROBOT_MODEL}. Two 6-DoF arms, LEFT and RIGHT, each with a parallel gripper.
Row layout {list(ACTION_NAMES)} at {CONTROL_HZ:.0f} Hz. Joints are ABSOLUTE
TARGET ANGLES IN RADIANS. Each gripper is absolute in {list(GRIPPER_RANGE)}
with 0 = CLOSED and 1 = OPEN.

Cameras: {', '.join(CAMERA_NAMES)} -- top is the scene view, left and right are
the wrist cameras of the matching arm.

Gripper values slightly outside [0,1] are an EXPECTED artefact of quantile
normalisation without clipping, not a fault in the proposal. Do not treat them as
evidence for a takeover.

MODES
  student - keep the chunk; 1-{MAX_STUDENT_STEPS} leading steps
  edit    - bounded JOINT-SPACE adjustment per arm in radians, 1-{MAX_CORRECTED_STEPS}
            leading steps, each joint within +/-{MAX_CORRECTION_RAD:.4f} rad.
            (Upstream's edit is Cartesian; this rig has no verified tool-tip
            transform or calibrated arm base frames, so corrections are in
            joint space.) Use zeros and "keep" for an arm you do not correct.
  eef     - you may return it, but it will be REFUSED by the execution gate on
            this robot. Prefer student or edit.
  stop    - the scene looks unsafe, or the task already appears complete."""

FK_NOTE = ("FK preview is robot-only forward kinematics of the commanded joint "
           "targets, per arm, in THAT ARM'S OWN BASE FRAME, at the gripper mount "
           "link. The two bases are not calibrated into one frame, so do not "
           "compare left and right positions. FK is not a simulation of "
           "contact, grasping, objects or future success.")

PROVENANCE_NOTE = {
    "recorded_demo": _KUKA_PROVENANCE["recorded_demo"],
    "model_predicted": (
        "PROVENANCE: the action chunk below is a pi0.5 MODEL PROPOSAL "
        f"({CHECKPOINT['repo_id']}) that has NOT been executed. No image shows "
        "its result, and nothing you can see is a consequence of it."),
    "astra_direct": _KUKA_PROVENANCE["astra_direct"],
}


def _fmt(values: Sequence[float], nd: int = 3) -> list[float]:
    return [round(float(v), nd) for v in values]


def build_packet(*, task_instruction: str, observation_id: str,
                 state: Sequence[float], chunk: Sequence[Sequence[float]],
                 provenance: str, frames: dict[str, Any],
                 fk_preview: dict[str, Any] | None = None,
                 history: dict[str, Any] | None = None,
                 reference: Any = None,
                 preview_steps: int = PREVIEW_STEPS) -> dict[str, Any]:
    """Assemble the review request. Returns a dict; sends nothing."""
    if provenance not in PROVENANCE_NOTE:
        raise ValueError(f"unknown provenance {provenance!r}; must be one of "
                         f"{sorted(PROVENANCE_NOTE)}")
    rows = [list(r) for r in chunk]
    head = rows[:preview_steps]
    now = split_arms(state)
    lines = [PROVENANCE_NOTE[provenance], ""]
    if reference is not None:
        lines += [reference.render(), ""]
    lines += [f"request_id: {observation_id}", f"task: {task_instruction}"]
    for arm in ARMS:
        lines.append(f"current_{arm}_joints_rad: {_fmt(now[arm]['joints'])}  "
                     f"gripper: {float(now[arm]['gripper']):.3f}")
    lines += [f"chunk: {len(rows)} steps @ {CONTROL_HZ:.0f} Hz "
              f"({len(rows) / CONTROL_HZ:.2f} s); first {len(head)} shown",
              "ABSOLUTE joint targets rad and gripper, per arm:"]
    for i, r in enumerate(head):
        a = split_arms(r)
        lines.append(f"  t+{i:02d}: L {_fmt(a['left']['joints'])} "
                     f"g={float(a['left']['gripper']):.3f} | "
                     f"R {_fmt(a['right']['joints'])} "
                     f"g={float(a['right']['gripper']):.3f}")
    if history:
        lines += ["", "RECORDED HISTORY (already executed, ground truth):",
                  f"  {history.get('summary', '')}"]
    if fk_preview and fk_preview.get("available"):
        traj = fk_preview.get("trajectory") or []
        lines += ["", FK_NOTE,
                  f"gripper-mount positions (m), first {min(len(traj), preview_steps)}:"]
        for p in traj[:preview_steps]:
            lines.append(f"  t+{p['step']:02d}: L {_fmt(p['left'], 4)} "
                         f"R {_fmt(p['right'], 4)}")
    lines += ["", "OBSERVATION FRAMES (open these; they are the visual evidence):"]
    for cam, f in sorted(frames.items()):
        lines.append(f"  {cam}: frame {f.get('frame_index')} "
                     f"{f.get('frame_path') or f.get('video') or '(live)'}")
    return {
        "schema": SCHEMA,
        "request_id": observation_id,
        "system": GATE_INSTRUCTION + "\n\n" + YAM_CONTRACT_NOTE,
        "user_text": "\n".join(lines),
        "frames": frames,
        "response_schema": response_schema(request_id=observation_id),
        "provenance": provenance,
        "reference": reference.to_log() if reference is not None else None,
        "gripper_indices": dict(GRIPPER_INDICES),
        "sends_nothing": True,
    }
