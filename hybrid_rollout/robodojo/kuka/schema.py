"""KUKA response schema, preserving the upstream decision contract.

Upstream `skill/schema.py` is the reference. The assessment block, the mode
enum, the step bounds and the field names are carried over verbatim so a
reviewer prompted with the unchanged gate produces a response this parses.

Two changes, both forced by the embodiment and neither optional:
  - upstream nests `edit`/`target` under {left, right} for a dual ARX-X5. One
    arm means those wrappers are dropped.
  - upstream `edit` is Cartesian (delta_position m, delta_rotation_vector rad).
    With no verified TCP and no IK, ours is JOINT-SPACE degrees.
`eef` REMAINS in the mode enum and in the parser. It is a legitimate upstream
decision and removing it would fork the decision contract, so the schema stays
compatible and the refusal happens later, at the KUKA execution gate
(contract.eef_execution_gate). An eef decision is parsed, recorded and then
deterministically refused; it never produces a robot command.
"""
from __future__ import annotations

from typing import Any

from .contract import ARM_DIM, MAX_CORRECTED_STEPS, MAX_STUDENT_STEPS

EXECUTION_STATUSES = ("not_started", "progressing", "failed", "uncertain", "recovered")
INTENT_STATUSES = ("aligned", "misaligned", "uncertain")
MODES = ("student", "edit", "eef", "stop")


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    """Upstream skill/schema.py `_obj`, unchanged: strict, all-required."""
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def response_schema(request_id: str | None = None,
                    include_eef: bool = True) -> dict[str, Any]:
    """Upstream-compatible. `include_eef` defaults True: the decision contract is
    preserved even though eef execution is gated off elsewhere."""
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
    modes = [m for m in MODES if m != "eef" or include_eef]
    props = dict(
        request_id=identifier,
        mode={"type": "string", "enum": modes},
        steps={"type": "integer", "minimum": 1, "maximum": MAX_STUDENT_STEPS},
        reason=string,
        edit=_obj(dict(
            delta_joint_deg={"type": "array", "items": number,
                             "minItems": ARM_DIM, "maxItems": ARM_DIM},
            gripper={"type": "string", "enum": ["keep", "open", "closed"]})),
        assessment=assessment)
    if include_eef:
        # Same shape as upstream skill/schema.py `target`, single-arm.
        props["target"] = _obj(dict(position=vector3, quaternion_wxyz=vector4,
                                    gripper_closed={"type": "boolean"}))
    return _obj(props)


def step_bounds(mode: str) -> tuple[int, int]:
    """Upstream SKILL.md: student 1-15, takeover 1-5."""
    return (1, MAX_CORRECTED_STEPS) if mode in ("edit", "eef") else (1, MAX_STUDENT_STEPS)
