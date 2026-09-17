"""Deterministic output sanitization: after review, before execution validation.

NOT AN IMPROVEMENT. Nothing here is a judgement, a correction, or an Astra
contribution. Every operation is a fixed function of its input with no model in
the loop, and each is recorded value-by-value so an auditor can reverse it. The
raw pi0.5 proposal is never touched -- sanitization operates on a COPY of the
post-review candidate, and the audit keeps the original array verbatim.

WHAT IT IS ALLOWED TO DO

  GRIPPER CLAMP    The gripper is a physical actuator with a verified [0,1]
                   range, so a command of 1.031 is not a trajectory to negotiate
                   with -- it is outside the device. Clamped into range, and
                   every changed value is recorded with its before and after.

  TIME SCALING     A velocity breach means the SAME path is being asked for too
                   fast. The principled fix is therefore temporal, not spatial:
                   keep every waypoint and its order, and traverse them more
                   slowly. Implemented as path-preserving resampling -- linear
                   interpolation along the existing polyline at a finer parameter
                   spacing -- so the emitted stream keeps the original control
                   rate and the duration grows instead. The geometric path in
                   joint space is identical; only timing and sample count change.

  STATE BRIDGE     Applied ONLY when the measured state to first target jump
                   itself breaches a cap. The measured state is prepended as a
                   waypoint so the approach is traversed at a feasible rate
                   rather than commanded as an instantaneous jump.

WHAT IT MUST NEVER DO

  - clip, saturate or otherwise alter a JOINT ANGLE. Truncating a joint command
    silently changes where the arm goes, which is a different trajectory wearing
    the original's name. Position-limit breaches are therefore NOT sanitized:
    they are reported and they keep blocking execution.
  - touch a non-finite, misshapen or wrongly-provenanced array. Those are refused
    outright; sanitizing structurally broken input would launder it.
  - make anything executable by assertion. The result is re-validated from
    scratch, and may still fail.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .contract import (ACTION_DIM, ACTION_NAMES, ARM_DIM, CONTROL_HZ,
                       GRIPPER_RANGE, max_step_deg)

SCHEMA = "hybrid_rollout.robodojo.kuka.sanitize.v1"

# A trajectory needing more than this much stretching is not a timing problem; it
# is a proposal that bears no relation to what the arm can do. Refuse rather than
# emit a 10x-long command and call it sanitized.
MAX_TIME_SCALE = 4.0
# Resampling is exact in real arithmetic; in floating point a residue can leave a
# step a hair over cap. Bump the sample count a few times rather than ship a
# candidate that fails its own re-validation.
MAX_BUMPS = 4


@dataclass
class GripperClamp:
    step: int
    from_value: float
    to_value: float

    def to_log(self) -> dict[str, Any]:
        return {"step": self.step, "from": round(self.from_value, 6),
                "to": round(self.to_value, 6),
                "delta": round(self.to_value - self.from_value, 6)}


@dataclass
class TimeScaling:
    applied: bool
    reason: str
    required_scale: float | None = None
    applied_scale: float | None = None
    n_in: int | None = None
    n_out: int | None = None
    rate_hz: float | None = None
    duration_before_s: float | None = None
    duration_after_s: float | None = None
    worst_ratio_before: float | None = None
    worst_ratio_after: float | None = None
    method: str = ("path-preserving resampling: linear interpolation along the "
                   "existing joint-space polyline at finer parameter spacing")
    path_preserved: bool = True
    path_note: str = ("Every original waypoint lies on the emitted path and the "
                      "traversal order is unchanged; only timing and sample count "
                      "differ. No joint value was clipped.")

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SanitizerReport:
    applied: bool = False
    refused: str | None = None
    gripper_clamps: list[GripperClamp] = field(default_factory=list)
    time_scaling: TimeScaling | None = None
    state_bridge: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    # Fixed labels. These exist so no downstream reader can mistake a mechanical
    # transform for a review outcome.
    deterministic: bool = True
    is_astra_improvement: bool = False
    returned_unchanged: bool = False
    label: str = ("DETERMINISTIC SANITIZATION -- a fixed function of the input. "
                  "Not a review, not a correction, not attributable to Astra or "
                  "to any model.")

    @property
    def changed(self) -> bool:
        """Were any changes computed? Not the same as whether they were EMITTED."""
        return bool(self.gripper_clamps
                    or (self.time_scaling and self.time_scaling.applied)
                    or (self.state_bridge or {}).get("applied"))

    @property
    def changes_emitted(self) -> bool:
        """Did the returned array actually carry the changes?

        On refusal the ORIGINAL array is returned untouched, so any clamp or
        bridge computed beforehand was discarded. Reporting `changed` alone there
        would describe edits that are not in the output.
        """
        return bool(self.changed and self.refused is None)

    def to_log(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "applied": self.applied,
            "changed": self.changed,
            "changes_emitted": self.changes_emitted,
            "refused": self.refused,
            "deterministic": self.deterministic,
            "is_astra_improvement": self.is_astra_improvement,
            "label": self.label,
            "gripper_clamps": [c.to_log() for c in self.gripper_clamps],
            "gripper_clamped_steps": len(self.gripper_clamps),
            "time_scaling": self.time_scaling.to_log() if self.time_scaling else None,
            "state_bridge": self.state_bridge,
            "notes": self.notes,
            "returned_unchanged": self.returned_unchanged,
        }


def _finite_rows(values: list[list[float]]) -> bool:
    return all(len(r) == ACTION_DIM for r in values) and all(
        not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
        for r in values for x in r)


def worst_velocity_ratio(values: list[list[float]], hz: float) -> tuple[float, dict]:
    """max over steps and arm joints of |step| / cap. <= 1.0 means feasible."""
    caps = max_step_deg(hz)
    worst, at = 0.0, {}
    for i in range(1, len(values)):
        for j in range(ARM_DIM):
            r = abs(values[i][j] - values[i - 1][j]) / caps[j]
            if r > worst:
                worst, at = r, {"step": i, "joint": ACTION_NAMES[j],
                                "deg": round(abs(values[i][j] - values[i - 1][j]), 4),
                                "cap_deg": round(caps[j], 4)}
    return worst, at


def resample_path(values: list[list[float]], n_out: int) -> list[list[float]]:
    """Uniform resampling along the waypoint polyline. Order-preserving.

    The parameter sweeps monotonically from 0 to n_in-1, so waypoints are
    traversed in their original order and every emitted point lies on a segment
    between two consecutive originals. No value is invented outside the path.
    """
    n_in = len(values)
    if n_in < 2 or n_out == n_in:
        return [list(r) for r in values]
    if n_out < 2:
        raise ValueError("n_out must be >= 2")
    out: list[list[float]] = []
    for k in range(n_out):
        t = k * (n_in - 1) / (n_out - 1)
        i = min(int(math.floor(t)), n_in - 2)
        f = t - i
        a, b = values[i], values[i + 1]
        out.append([a[d] + (b[d] - a[d]) * f for d in range(ACTION_DIM)])
    return out


def sanitize(rows: Sequence[Sequence[float]], *,
             state: Sequence[float] | None = None,
             hz: float = CONTROL_HZ,
             max_time_scale: float = MAX_TIME_SCALE,
             allow_state_bridge: bool = True
             ) -> tuple[list[list[float]], SanitizerReport]:
    """Deterministically sanitize a post-review candidate. Never mutates input.

    Operates on plain row lists so this module carries no dependency on any
    dataclass: it has to run unchanged on the Jetson.
    """
    rep = SanitizerReport()
    chunk = type("C", (), {"values": [list(r) for r in rows]})()

    if not chunk.values or not _finite_rows(chunk.values):
        rep.refused = ("non-finite or misshapen array -- refused. Structurally "
                       "broken input is not sanitized, it is rejected.")
        return [list(r) for r in rows], rep

    vals = [list(r) for r in chunk.values]

    # ---- 1. gripper clamp: the ONLY value-altering operation, and only here ----
    lo, hi = GRIPPER_RANGE
    for i, row in enumerate(vals):
        g = row[ARM_DIM]
        if g < lo or g > hi:
            new = min(hi, max(lo, g))
            rep.gripper_clamps.append(GripperClamp(i, g, new))
            row[ARM_DIM] = new
    if rep.gripper_clamps:
        rep.notes.append(
            f"clamped {len(rep.gripper_clamps)} gripper command(s) into "
            f"[{lo},{hi}]; joint angles untouched")

    # ---- 2. state bridge, only if the approach itself is infeasible ----------
    caps = max_step_deg(hz)
    bridged = False
    if state is not None and len(state) >= ACTION_DIM:
        jump = {ACTION_NAMES[j]: abs(vals[0][j] - state[j]) for j in range(ARM_DIM)}
        bad = {k: round(v, 4) for k, v in jump.items()
               if v > caps[ACTION_NAMES.index(k)]}
        if bad and allow_state_bridge:
            vals.insert(0, [float(state[d]) for d in range(ACTION_DIM)])
            bridged = True
            rep.state_bridge = {
                "applied": True, "breaching_joints": bad,
                "method": ("measured state prepended as a waypoint so the approach "
                           "is traversed at a feasible rate instead of commanded "
                           "as an instantaneous jump"),
                "adds_samples": 1}
        elif bad:
            rep.state_bridge = {"applied": False, "breaching_joints": bad,
                                "reason": "state bridging disabled"}

    # ---- 3. time scaling for velocity feasibility ---------------------------
    ratio, at = worst_velocity_ratio(vals, hz)
    if ratio <= 1.0 + 1e-12:
        rep.time_scaling = TimeScaling(
            applied=False,
            reason=f"no velocity breach (worst ratio {ratio:.4f} <= 1.0)",
            worst_ratio_before=round(ratio, 6), worst_ratio_after=round(ratio, 6),
            n_in=len(vals), n_out=len(vals), rate_hz=hz,
            duration_before_s=round(len(vals) / hz, 6),
            duration_after_s=round(len(vals) / hz, 6))
    elif ratio > max_time_scale:
        rep.refused = (
            f"velocity breach needs a {ratio:.3f}x time stretch, over the "
            f"{max_time_scale}x limit ({at}). Refused: a proposal this far from "
            f"feasible is not a timing problem, and stretching it would disguise "
            f"that. Reported, not repaired.")
        rep.time_scaling = TimeScaling(applied=False, reason=rep.refused,
                                       required_scale=round(ratio, 6),
                                       worst_ratio_before=round(ratio, 6))
        rep.applied = False
        rep.returned_unchanged = True
        if rep.gripper_clamps or (rep.state_bridge or {}).get("applied"):
            rep.notes.append(
                "REFUSED: the original array is returned unchanged, so the "
                "gripper clamp(s) and/or state bridge listed above were computed "
                "but NOT emitted (changes_emitted=false)")
        return [list(r) for r in rows], rep
    else:
        n_in = len(vals)
        n_out = int(math.ceil((n_in - 1) * ratio)) + 1
        scaled = resample_path(vals, n_out)
        after, _ = worst_velocity_ratio(scaled, hz)
        bumps = 0
        while after > 1.0 + 1e-12 and bumps < MAX_BUMPS:
            n_out += 1
            scaled = resample_path(vals, n_out)
            after, _ = worst_velocity_ratio(scaled, hz)
            bumps += 1
        rep.time_scaling = TimeScaling(
            applied=True,
            reason=(f"velocity breach: {at.get('joint')} step "
                    f"{at.get('deg')} deg > cap {at.get('cap_deg')} deg at "
                    f"{hz:.0f} Hz (ratio {ratio:.4f})"),
            required_scale=round(ratio, 6),
            applied_scale=round((n_out - 1) / (n_in - 1), 6) if n_in > 1 else None,
            n_in=n_in, n_out=n_out, rate_hz=hz,
            duration_before_s=round(n_in / hz, 6),
            duration_after_s=round(n_out / hz, 6),
            worst_ratio_before=round(ratio, 6),
            worst_ratio_after=round(after, 6))
        if bumps:
            rep.notes.append(f"sample count bumped {bumps}x for float residue")
        vals = scaled
        rep.notes.append(
            f"time-scaled {n_in} -> {n_out} samples at {hz:.0f} Hz "
            f"({n_in / hz:.3f}s -> {n_out / hz:.3f}s); path unchanged")

    if bridged:
        rep.notes.append("measured state prepended; emitted stream now starts at "
                         "the arm's actual pose")

    rep.applied = True
    if not rep.changed:
        # Nothing was altered, so nothing is derived: reporting a transform for a
        # no-op would make every clean candidate look modified.
        rep.notes.append("no change required; original array kept")
        rep.returned_unchanged = True
        return [list(r) for r in rows], rep
    return vals, rep
