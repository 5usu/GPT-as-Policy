"""Building the Astra review packet, on the UNCHANGED upstream gate.

The gate instruction is imported from `robodojo_server/gate_assessment.py` and is
never edited or paraphrased here. Rewriting it would fork the thing the upstream
result rests on; what this module does is assemble the KUKA-specific evidence
around it.

WHAT GOES IN, AND WHY EACH PIECE
  the gate         upstream GATE_INSTRUCTION, verbatim
  the contract     units, rates, absolute-joint semantics, and the known
                   normalisation overshoot, so the reviewer does not mistake an
                   artefact for a defect
  provenance       whether the chunk under review is a RECORDED DEMONSTRATION or
                   a MODEL PROPOSAL. These have completely different standing and
                   conflating them would invalidate the review.
  FK preview       flange-frame only, explicitly not tool-tip and explicitly not
                   a prediction about objects
  frames           referenced by path; this module never loads or embeds images

NOTHING HERE CALLS AN API. `build_packet` returns a dict; sending it is the
caller's decision and requires its own explicit approval.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..robodojo_server.gate_assessment import GATE_INSTRUCTION  # upstream, verbatim
from .contract import (ACTION_NAMES, ARM_DIM, CONTROL_HZ, GRIPPER_RANGE,
                       MAX_CORRECTION_DEG, MAX_CORRECTED_STEPS, MAX_STUDENT_STEPS,
                       ROBOT_MODEL)
from .schema import response_schema

SCHEMA = "hybrid_rollout.robodojo.kuka.packet.v1"
PREVIEW_STEPS = 10

KUKA_CONTRACT_NOTE = f"""ROBOT AND ACTION CONTRACT
{ROBOT_MODEL}, one 6-DoF arm. Dimensions {list(ACTION_NAMES)} at {CONTROL_HZ:.0f} Hz.
A1-A6 are ABSOLUTE JOINT TARGET ANGLES IN DEGREES; gripper is absolute in
{list(GRIPPER_RANGE)}. There is no second arm; ignore any dual-arm convention.

Gripper values slightly outside [0,1] are an EXPECTED artefact of quantile
normalisation without clipping, not a fault in the proposal. Do not treat them as
evidence for a takeover.

MODES
  student - keep the chunk; 1-{MAX_STUDENT_STEPS} leading steps
  edit    - bounded JOINT-SPACE adjustment in degrees, 1-{MAX_CORRECTED_STEPS}
            leading steps, each joint within +/-{MAX_CORRECTION_DEG} deg.
            (Upstream's edit is Cartesian; this arm has no verified TCP or IK,
            so corrections are expressed in joint space.)
  eef     - you may return it, but it will be REFUSED by the execution gate on
            this robot until the tool transform, IK, workspace and collision
            model are all verified. Prefer student or edit.
  stop    - the scene looks unsafe, or the task already appears complete."""

FK_NOTE = ("FK preview is robot-only forward kinematics of the commanded joint "
           "targets, in the FLANGE frame. The custom tool transform is "
           "unverified, so these are NOT tool-tip positions. FK is not a "
           "simulation of contact, grasping, objects or future success.")

PROVENANCE_NOTE = {
    "recorded_demo": (
        "PROVENANCE: the action chunk below is a RECORDED HUMAN DEMONSTRATION "
        "that was actually executed on this robot. The observation frames are "
        "evidence of what happened. You are assessing what a successful episode "
        "looks like, not correcting a model."),
    "model_predicted": (
        "PROVENANCE: the action chunk below is a pi0.5 MODEL PROPOSAL that has "
        "NOT been executed. No image shows its result, and nothing you can see "
        "is a consequence of it."),
    "astra_direct": (
        "PROVENANCE: you are proposing the action yourself, within the bounded "
        "action space given. It has not been executed."),
}


def build_packet(*, task_instruction: str, observation_id: str,
                 state: Sequence[float], chunk: Sequence[Sequence[float]],
                 provenance: str, frames: dict[str, Any],
                 fk_preview: dict[str, Any] | None = None,
                 history: dict[str, Any] | None = None,
                 preview_steps: int = PREVIEW_STEPS) -> dict[str, Any]:
    """Assemble the review request. Returns a dict; sends nothing."""
    if provenance not in PROVENANCE_NOTE:
        raise ValueError(f"unknown provenance {provenance!r}; must be one of "
                         f"{sorted(PROVENANCE_NOTE)}")
    rows = [list(r) for r in chunk]
    head = rows[:preview_steps]
    lines = [
        PROVENANCE_NOTE[provenance], "",
        f"request_id: {observation_id}",
        f"task: {task_instruction}",
        f"current_joints_deg: {[round(v, 2) for v in list(state)[:ARM_DIM]]}",
        f"current_gripper: {round(float(list(state)[ARM_DIM]), 3)}",
        f"chunk: {len(rows)} steps @ {CONTROL_HZ:.0f} Hz "
        f"({len(rows) / CONTROL_HZ:.2f} s); first {len(head)} shown",
        "ABSOLUTE joint targets deg (A1..A6) and gripper:",
    ]
    for i, r in enumerate(head):
        lines.append(f"  t+{i:02d}: {[round(v, 3) for v in r[:ARM_DIM]]} "
                     f"grip={float(r[ARM_DIM]):.3f}")
    if history:
        lines += ["", "RECORDED HISTORY (already executed, ground truth):",
                  f"  {history.get('summary', '')}"]
    if fk_preview and fk_preview.get("available"):
        traj = fk_preview.get("trajectory") or []
        lines += ["", FK_NOTE,
                  f"flange positions (m), first {min(len(traj), preview_steps)}:"]
        for p in traj[:preview_steps]:
            lines.append(f"  t+{p['step']:02d}: "
                         f"{[round(v, 4) for v in p['position']]}")
    lines += ["", "OBSERVATION FRAMES (open these; they are the visual evidence):"]
    for cam, f in sorted(frames.items()):
        lines.append(f"  {cam}: frame {f.get('frame_index')} "
                     f"{f.get('frame_path') or f.get('video') or '(unresolved)'}")
    return {
        "schema": SCHEMA,
        "request_id": observation_id,
        "system": GATE_INSTRUCTION + "\n\n" + KUKA_CONTRACT_NOTE,
        "user_text": "\n".join(lines),
        "frames": frames,
        "response_schema": response_schema(request_id=observation_id),
        "provenance": provenance,
        "sends_nothing": True,
    }
