"""Deterministic KUKA validation. Stdlib only -- must run on the Jetson.

Ported from the tested offline implementation and adapted to the upstream flow.

TWO SEPARATE JUDGEMENTS, NEVER CONFLATED
  improvement_valid  comparative: did an EDIT introduce or worsen anything?
                     Inherited problems are permitted here; the question is
                     about the edit.
  execution_safe     absolute and FAIL-CLOSED: does the FINAL candidate breach
                     any hard limit? Inherited counts. The pi0.5 proposal's own
                     6.992 deg A2 step is unexecutable at 30 Hz regardless of
                     who authored it or whether an edit touched it.

A candidate may legitimately be improvement_valid=True and execution_safe=False.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .contract import (ACTION_DIM, ACTION_NAMES, ARM_DIM, CONTROL_HZ,
                       DATA_ENVELOPE_MAX_DEG, DATA_ENVELOPE_MIN_DEG,
                       DATA_ENVELOPE_SLACK_DEG, GRIPPER_RANGE,
                       POSITION_LIMIT_DEG, max_step_deg)


class Severity(str, Enum):
    FATAL = "fatal"          # structurally broken or unreachable; emit nothing
    LIMIT = "limit"          # a real controller-limit breach
    ADVISORY = "advisory"    # known artefact; explained, still blocks execution


@dataclass(frozen=True)
class Violation:
    code: str
    severity: Severity
    detail: str
    joint: str | None = None
    magnitude: float | None = None
    limit: float | None = None

    def key(self) -> tuple[str, str | None]:
        return (self.code, self.joint)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


# A breach of ANY of these makes a candidate ineligible for execution, whatever
# its severity class and whoever authored it. `gripper_out_of_range` is ADVISORY
# because its CAUSE is a known quantile-normalisation artefact -- useful to say
# -- but an out-of-range command is still out of range.
HARD_VIOLATION_CODES = frozenset({
    "provenance_mismatch", "bad_shape", "non_finite", "position_limit",
    "velocity_limit", "state_discontinuity", "gripper_out_of_range"})


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

    for j in range(ARM_DIM):
        lo_l, hi_l = POSITION_LIMIT_DEG[j]
        lo = min(r[j] for r in rows); hi = max(r[j] for r in rows)
        if lo < lo_l or hi > hi_l:
            v.append(Violation("position_limit", Severity.FATAL,
                               f"range [{lo:.3f}, {hi:.3f}] outside [{lo_l}, {hi_l}]",
                               joint=ACTION_NAMES[j],
                               magnitude=round(max(lo_l - lo, hi - hi_l), 4),
                               limit=hi_l))

    caps = max_step_deg(hz)
    for j in range(ARM_DIM):
        worst, at = 0.0, None
        for i in range(1, len(rows)):
            d = abs(rows[i][j] - rows[i - 1][j])
            if d > worst:
                worst, at = d, i
        if worst > caps[j]:
            v.append(Violation("velocity_limit", Severity.LIMIT,
                               f"step {at}: {worst:.4f} deg > cap {caps[j]:.4f} "
                               f"deg/step at {hz:.0f} Hz",
                               joint=ACTION_NAMES[j], magnitude=round(worst, 4),
                               limit=round(caps[j], 4)))

    if state is not None and len(state) >= ACTION_DIM:
        for j in range(ARM_DIM):
            d = abs(rows[0][j] - state[j])
            if d > caps[j]:
                v.append(Violation("state_discontinuity", Severity.LIMIT,
                                   f"state -> row 0 jump {d:.4f} deg > cap "
                                   f"{caps[j]:.4f}", joint=ACTION_NAMES[j],
                                   magnitude=round(d, 4), limit=round(caps[j], 4)))

    lo_g, hi_g = GRIPPER_RANGE
    bad = [i for i, r in enumerate(rows) if not (lo_g <= r[ARM_DIM] <= hi_g)]
    if bad:
        worst = max(max(lo_g - rows[i][ARM_DIM], rows[i][ARM_DIM] - hi_g)
                    for i in bad)
        v.append(Violation("gripper_out_of_range", Severity.ADVISORY,
                           f"{len(bad)}/{len(rows)} steps outside [{lo_g},{hi_g}], "
                           f"worst {worst:.4f} -- expected normalisation artefact",
                           joint="gripper", magnitude=round(worst, 6), limit=hi_g))

    if check_envelope:
        for j in range(ARM_DIM):
            lo = min(r[j] for r in rows); hi = max(r[j] for r in rows)
            if lo < DATA_ENVELOPE_MIN_DEG[j] - DATA_ENVELOPE_SLACK_DEG or \
                    hi > DATA_ENVELOPE_MAX_DEG[j] + DATA_ENVELOPE_SLACK_DEG:
                v.append(Violation("outside_training_envelope", Severity.ADVISORY,
                                   f"[{lo:.2f}, {hi:.2f}] leaves the observed "
                                   f"corpus range; a sanity signal, NOT a safety "
                                   f"limit", joint=ACTION_NAMES[j]))
    return v


def has_fatal(vs: Iterable[Violation]) -> bool:
    return any(x.severity is Severity.FATAL for x in vs)


def execution_eligible(vs: list[Violation] | None,
                       emitted: bool = True) -> tuple[bool, list[str]]:
    """FAIL-CLOSED. Eligible only if something was emitted, validation ran, and
    no hard limit is breached."""
    if not emitted or vs is None:
        return False, ["nothing emitted or validation not run (fail-closed)"]
    blockers = sorted({f"{x.code}" + (f"[{x.joint}]" if x.joint else "")
                       for x in vs if x.code in HARD_VIOLATION_CODES})
    return (not blockers), blockers


def worsens(before: list[Violation],
            after: list[Violation]) -> tuple[bool, str]:
    """Did an edit introduce a NEW violation or make an inherited one worse?"""
    b = {x.key(): x for x in before}
    for a in after:
        if a.key() not in b:
            return True, (f"introduced {a.code}"
                          + (f" on {a.joint}" if a.joint else ""))
        prev = b[a.key()]
        if (a.magnitude is not None and prev.magnitude is not None
                and a.magnitude > prev.magnitude + 1e-9):
            return True, (f"worsened {a.code}"
                          + (f" on {a.joint}" if a.joint else "")
                          + f": {prev.magnitude} -> {a.magnitude}")
    return False, "no new or worsened violation"


def summarize(vs: list[Violation]) -> dict[str, Any]:
    return {"n": len(vs),
            "by_severity": {s.value: sum(1 for x in vs if x.severity is s)
                            for s in Severity},
            "codes": sorted({x.code for x in vs}),
            "items": [x.to_log() for x in vs]}
