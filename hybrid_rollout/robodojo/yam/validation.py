"""Deterministic YAM validation. Stdlib only.

Same rules and the same two judgements as `kuka.validation`, run over both
arms: `improvement_valid` asks whether an edit made anything worse,
`execution_safe` asks -- fail-closed -- whether the final candidate breaches a
hard limit, inherited or not. The Violation type, the severity classes and the
comparison helpers are the KUKA ones, unchanged, so an audit row reads the
same on either robot.
"""
from __future__ import annotations

import math
from typing import Sequence

from ..kuka.validation import (HARD_VIOLATION_CODES, Severity, Violation,
                               execution_eligible, has_fatal, summarize, worsens)
from .contract import (ACTION_DIM, ACTION_NAMES, CONTROL_HZ,
                       DATA_ENVELOPE_Q01, DATA_ENVELOPE_Q99,
                       DATA_ENVELOPE_SLACK_RAD, GRIPPER_INDICES, GRIPPER_RANGE,
                       JOINT_INDICES, arm_joint_limit, max_step_rad)

__all__ = ["HARD_VIOLATION_CODES", "Severity", "Violation", "execution_eligible",
           "has_fatal", "summarize", "validate_chunk", "worsens"]


def validate_chunk(rows: Sequence[Sequence[float]], *,
                   state: Sequence[float] | None = None,
                   hz: float = CONTROL_HZ,
                   expect_provenance: str | None = None,
                   provenance: str | None = None,
                   check_envelope: bool = True) -> list[Violation]:
    """Position, velocity, gripper, continuity, shape/finite and provenance."""
    v: list[Violation] = []
    if expect_provenance is not None and provenance != expect_provenance:
        v.append(Violation("provenance_mismatch", Severity.FATAL,
                           f"expected {expect_provenance!r}, got {provenance!r}"))
    rows = [list(r) for r in rows]
    if not rows:
        v.append(Violation("bad_shape", Severity.FATAL, "empty chunk"))
        return v
    for i, row in enumerate(rows):
        if len(row) != ACTION_DIM:
            v.append(Violation("bad_shape", Severity.FATAL,
                               f"row {i} has {len(row)} dims, expected {ACTION_DIM}"))
            return v
        for j, x in enumerate(row):
            if isinstance(x, bool) or not isinstance(x, (int, float)) \
                    or not math.isfinite(x):
                v.append(Violation("non_finite", Severity.FATAL,
                                   f"row {i} {ACTION_NAMES[j]} = {x!r}",
                                   joint=ACTION_NAMES[j]))
                return v

    for j in JOINT_INDICES:
        lo_l, hi_l = arm_joint_limit(j)
        lo = min(r[j] for r in rows); hi = max(r[j] for r in rows)
        if lo < lo_l or hi > hi_l:
            v.append(Violation("position_limit", Severity.FATAL,
                               f"range [{lo:.4f}, {hi:.4f}] outside "
                               f"[{lo_l:.4f}, {hi_l:.4f}] rad",
                               joint=ACTION_NAMES[j],
                               magnitude=round(max(lo_l - lo, hi - hi_l), 5),
                               limit=hi_l))

    cap = max_step_rad(hz)
    for j in JOINT_INDICES:
        worst, at = 0.0, None
        for i in range(1, len(rows)):
            d = abs(rows[i][j] - rows[i - 1][j])
            if d > worst:
                worst, at = d, i
        if worst > cap:
            v.append(Violation("velocity_limit", Severity.LIMIT,
                               f"step {at}: {worst:.5f} rad > cap {cap:.5f} "
                               f"rad/step at {hz:.0f} Hz",
                               joint=ACTION_NAMES[j], magnitude=round(worst, 5),
                               limit=round(cap, 5)))

    if state is not None and len(state) >= ACTION_DIM:
        for j in JOINT_INDICES:
            d = abs(rows[0][j] - state[j])
            if d > cap:
                v.append(Violation("state_discontinuity", Severity.LIMIT,
                                   f"state -> row 0 jump {d:.5f} rad > cap "
                                   f"{cap:.5f}", joint=ACTION_NAMES[j],
                                   magnitude=round(d, 5), limit=round(cap, 5)))

    lo_g, hi_g = GRIPPER_RANGE
    for arm, g in GRIPPER_INDICES.items():
        bad = [i for i, r in enumerate(rows) if not (lo_g <= r[g] <= hi_g)]
        if bad:
            worst = max(max(lo_g - rows[i][g], rows[i][g] - hi_g) for i in bad)
            v.append(Violation("gripper_out_of_range", Severity.ADVISORY,
                               f"{len(bad)}/{len(rows)} steps outside "
                               f"[{lo_g},{hi_g}], worst {worst:.4f} -- expected "
                               f"quantile-normalisation artefact",
                               joint=ACTION_NAMES[g], magnitude=round(worst, 6),
                               limit=hi_g))

    if check_envelope:
        for j in JOINT_INDICES:
            lo = min(r[j] for r in rows); hi = max(r[j] for r in rows)
            if lo < DATA_ENVELOPE_Q01[j] - DATA_ENVELOPE_SLACK_RAD or \
                    hi > DATA_ENVELOPE_Q99[j] + DATA_ENVELOPE_SLACK_RAD:
                v.append(Violation("outside_training_envelope", Severity.ADVISORY,
                                   f"[{lo:.3f}, {hi:.3f}] leaves the checkpoint's "
                                   f"q01/q99 action range; a sanity signal, NOT "
                                   f"a safety limit", joint=ACTION_NAMES[j]))
    return v
