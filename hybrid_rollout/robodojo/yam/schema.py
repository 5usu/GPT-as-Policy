"""YAM response schema: the upstream decision contract, left/right intact.

Upstream `skill/schema.py` was written for a dual ARX-X5, so unlike the KUKA
port the `{left, right}` wrappers on `edit` and `target` stay exactly where
upstream put them. The assessment block, the mode enum, the step bounds and
the field names are carried over verbatim.

One change, forced by the rig and not optional: upstream `edit` is Cartesian
(delta_position m, delta_rotation_vector rad). With no verified tool-tip
transform and no calibrated frame joining the two arm bases, a Cartesian delta
cannot be turned into joint targets that anyone has checked, so each arm's edit
is JOINT-SPACE radians. `eef` stays in the enum and the parser, as on the KUKA;
it is refused later, at the YAM execution gate.
"""
from __future__ import annotations

from typing import Any

from ..kuka.schema import EXECUTION_STATUSES, INTENT_STATUSES, MODES, _obj
from .contract import JOINTS_PER_ARM, MAX_CORRECTED_STEPS, MAX_STUDENT_STEPS


def response_schema(request_id: str | None = None,
                    include_eef: bool = True) -> dict[str, Any]:
    number = {"type": "number"}
    string = {"type": "string"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}

    progress = _obj(dict(
        verified_completed={"type": "array", "items": string},
        currently_attempting=string,
        remaining={"type": "array", "items": string}))
    assessment = _obj(dict(
        task_progress=progress,
        current_subgoal=string,
        execution_status={"type": "string", "enum": list(EXECUTION_STATUSES)},
        execution_evidence=string,
        expected_next_intent=string,
        predicted_next_intent=string,
        intent_status={"type": "string", "enum": list(INTENT_STATUSES)},
        intent_evidence=string))

    identifier = string if request_id is None else {"type": "string",
                                                    "enum": [request_id]}
    arm_edit = _obj(dict(
        delta_joint_rad={"type": "array", "items": number,
                         "minItems": JOINTS_PER_ARM, "maxItems": JOINTS_PER_ARM},
        gripper={"type": "string", "enum": ["keep", "open", "closed"]}))
    props = dict(
        request_id=identifier,
        mode={"type": "string", "enum": [m for m in MODES
                                         if m != "eef" or include_eef]},
        steps={"type": "integer", "minimum": 1, "maximum": MAX_STUDENT_STEPS},
        reason=string,
        edit=_obj(dict(left=arm_edit, right=arm_edit)),
        assessment=assessment)
    if include_eef:
        arm_target = _obj(dict(position=vector3, quaternion_wxyz=vector4,
                               gripper_closed={"type": "boolean"}))
        props["target"] = _obj(dict(left=arm_target, right=arm_target))
    return _obj(props)


def step_bounds(mode: str) -> tuple[int, int]:
    """Upstream SKILL.md: student 1-15, takeover 1-5."""
    return (1, MAX_CORRECTED_STEPS) if mode in ("edit", "eef") else (1, MAX_STUDENT_STEPS)
