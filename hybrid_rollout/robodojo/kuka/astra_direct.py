"""Astra-direct in JOINT space -- Astra proposes the action itself.

HOW THIS DIFFERS FROM UPSTREAM, AND WHY THAT IS DELIBERATE
Upstream's Direct evaluation is Cartesian-only: `gpt_only_client.py` refuses
joint actions outright ("GPT-only supports bounded EEF actions only; no
joint/student/edit/stop"). That is a property of THEIR benchmark, whose simulator
consumed end-effector poses. It is not a property of this cell.

This controller consumes joint corrections -- AK.A1..A6 -- and that is a
perfectly good channel for a self-proposed action. So Astra-direct here is
JOINT-SPACE, and it is reachable through the interface exactly as it stands. No
RKorr, no $TOOL calibration, no controller change.

WHAT THAT COSTS, STATED PLAINLY
  - This is NOT upstream's Direct, so their 26% success figure does not transfer.
    We would be measuring a different thing and must report it as such.
  - There is no pi0.5 proposal underneath. In the hybrid loop the policy provides
    a sanity floor: Astra choosing a prefix of something a trained policy
    produced is bounded by that policy's competence. Here there is no floor, and
    the only thing between a bad number and the arm is this module's bounds.

That second point is why every bound below is required and none is defaulted.

WHAT IS BOUNDED
  - per-step displacement from the MEASURED pose, per joint
  - total excursion across the whole proposal
  - number of steps
  - the training envelope, as a sanity signal
Everything is measured against the arm's ACTUAL pose, not against a previous
proposal, so a sequence of individually-small steps cannot walk somewhere far.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .contract import (ACTION_NAMES, ARM_DIM, DATA_ENVELOPE_MAX_DEG,
                       DATA_ENVELOPE_MIN_DEG, DATA_ENVELOPE_SLACK_DEG,
                       POSITION_LIMIT_DEG)

SCHEMA = "hybrid_rollout.robodojo.kuka.astra_direct.v1"

MODE = "astra_direct_joint"      # never "eef"; never upstream's "astra_direct"

UPSTREAM_DIVERGENCE = (
    "Upstream's Direct evaluation is Cartesian-only and refuses joint actions. "
    "This mode is JOINT-SPACE, chosen because this controller accepts AK.A1..A6 "
    "and does not accept RKorr. Results are therefore NOT comparable with "
    "upstream's Direct numbers and must not be reported as such.")


class DirectBoundsMissing(Exception):
    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "astra-direct joint bounds absent from deployment configuration: "
            + ", ".join(self.missing)
            + ". Refusing. Direct proposal removes the policy's sanity floor, so "
              "these bounds are the only thing between a bad number and the arm.")


@dataclass(frozen=True)
class DirectBounds:
    """Required bounds. Nothing here has a default."""
    max_step_deg: tuple[float, ...]        # per joint, per step, from MEASURED
    max_total_excursion_deg: tuple[float, ...]
    max_steps: int
    require_envelope: bool = True
    source: str = "deployment_config"

    @staticmethod
    def _vec(cfg: dict[str, Any], key: str, missing: list[str]):
        v = cfg.get(key)
        if v in (None, "", [], {}):
            missing.append(key)
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return tuple(float(v) for _ in range(ARM_DIM))
        seq = list(v)
        if len(seq) != ARM_DIM:
            missing.append(f"{key} (need {ARM_DIM} values)")
            return None
        for i, x in enumerate(seq):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0:
                missing.append(f"{key}[{i}] must be positive")
                return None
        return tuple(float(x) for x in seq)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DirectBounds":
        missing: list[str] = []
        step = cls._vec(cfg, "direct_max_step_deg", missing)
        total = cls._vec(cfg, "direct_max_total_excursion_deg", missing)
        n = cfg.get("direct_max_steps")
        if n in (None, "", [], {}):
            missing.append("direct_max_steps")
        elif not isinstance(n, int) or isinstance(n, bool) or n < 1:
            missing.append("direct_max_steps must be a positive integer")
        if missing:
            raise DirectBoundsMissing(missing)
        return cls(step, total, int(n),
                   bool(cfg.get("direct_require_envelope", True)),
                   str(cfg.get("source", "deployment_config")))

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"schema": SCHEMA, "mode": MODE,
                  "upstream_divergence": UPSTREAM_DIVERGENCE})
        return d


@dataclass
class DirectViolation:
    code: str
    detail: str
    joint: str | None = None
    magnitude: float | None = None
    limit: float | None = None

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


def validate_direct(rows: Sequence[Sequence[float]], *,
                    measured: Sequence[float],
                    bounds: DirectBounds) -> list[DirectViolation]:
    """Bound a self-proposed joint trajectory. FAIL-CLOSED."""
    v: list[DirectViolation] = []
    if not rows:
        return [DirectViolation("empty", "no rows proposed")]
    if len(rows) > bounds.max_steps:
        v.append(DirectViolation("too_many_steps",
                                 f"{len(rows)} steps > limit {bounds.max_steps}",
                                 magnitude=len(rows), limit=bounds.max_steps))
    if measured is None or len(measured) < ARM_DIM:
        return [DirectViolation("no_measured_pose",
                                "no measured joint pose; refusing to bound a "
                                "self-proposed action against an unknown pose")]
    m = [float(x) for x in list(measured)[:ARM_DIM]]

    prev = m
    for i, row in enumerate(rows):
        r = list(row)
        if len(r) < ARM_DIM:
            v.append(DirectViolation("bad_shape", f"row {i} has {len(r)} values"))
            return v
        for j in range(ARM_DIM):
            x = r[j]
            if isinstance(x, bool) or not isinstance(x, (int, float)) \
                    or not math.isfinite(x):
                v.append(DirectViolation("non_finite", f"row {i} {ACTION_NAMES[j]}",
                                         ACTION_NAMES[j]))
                return v
            lo, hi = POSITION_LIMIT_DEG[j]
            if not lo <= x <= hi:
                v.append(DirectViolation("position_limit",
                                         f"row {i}: {x:.3f} outside [{lo}, {hi}]",
                                         ACTION_NAMES[j], round(x, 4), hi))
            step = abs(x - prev[j])
            if step > bounds.max_step_deg[j]:
                v.append(DirectViolation(
                    "step_too_large",
                    f"row {i}: {step:.4f} deg from the previous point > "
                    f"{bounds.max_step_deg[j]}", ACTION_NAMES[j], round(step, 4),
                    bounds.max_step_deg[j]))
            # Excursion is measured from the ARM'S ACTUAL POSE, so a run of
            # individually-small steps cannot walk somewhere far.
            total = abs(x - m[j])
            if total > bounds.max_total_excursion_deg[j]:
                v.append(DirectViolation(
                    "excursion_too_large",
                    f"row {i}: {total:.4f} deg from the measured pose > "
                    f"{bounds.max_total_excursion_deg[j]}", ACTION_NAMES[j],
                    round(total, 4), bounds.max_total_excursion_deg[j]))
            if bounds.require_envelope:
                elo = DATA_ENVELOPE_MIN_DEG[j] - DATA_ENVELOPE_SLACK_DEG
                ehi = DATA_ENVELOPE_MAX_DEG[j] + DATA_ENVELOPE_SLACK_DEG
                if not elo <= x <= ehi:
                    v.append(DirectViolation(
                        "outside_training_envelope",
                        f"row {i}: {x:.3f} outside the observed corpus range "
                        f"[{elo:.2f}, {ehi:.2f}]", ACTION_NAMES[j]))
        prev = r
    return v


def direct_eligible(violations: Sequence[DirectViolation]) -> tuple[bool, list[str]]:
    return (not violations), sorted({x.code for x in violations})


def response_schema(request_id: str | None = None) -> dict[str, Any]:
    """Astra proposes ABSOLUTE joint targets. Deliberately not upstream's shape."""
    from .schema import _obj
    number = {"type": "number"}
    string = {"type": "string"}
    ident = string if request_id is None else {"type": "string",
                                               "enum": [request_id]}
    return _obj(dict(
        request_id=ident,
        mode={"type": "string", "enum": [MODE, "stop"]},
        steps={"type": "integer", "minimum": 1, "maximum": 5},
        reason=string,
        joint_targets_deg={
            "type": "array",
            "items": {"type": "array", "items": number,
                      "minItems": ARM_DIM, "maxItems": ARM_DIM}},
        assessment=_obj(dict(
            task_progress=_obj(dict(
                verified_completed={"type": "array", "items": string},
                currently_attempting=string,
                remaining={"type": "array", "items": string})),
            current_subgoal=string,
            execution_status={"type": "string", "enum": [
                "not_started", "progressing", "failed", "uncertain", "recovered"]},
            execution_evidence=string,
            expected_next_intent=string,
            predicted_next_intent=string,
            intent_status={"type": "string",
                           "enum": ["aligned", "misaligned", "uncertain"]},
            intent_evidence=string)),
    ))
