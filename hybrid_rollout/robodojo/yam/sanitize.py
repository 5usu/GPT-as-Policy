"""Deterministic output sanitization for YAM, after review, before validation.

The rules are `kuka.sanitize`'s, word for word, and so are the report types:
clamp the grippers into [0, 1], stretch time along the unchanged joint-space
path when a velocity cap is breached, prepend the measured state when the
approach itself is infeasible -- and NEVER clip a joint angle. The only change
is that "the arm" is now both arms, so every rule runs over JOINT_INDICES and
GRIPPER_INDICES instead of range(6) and index 6.
"""
from __future__ import annotations

import math
from typing import Sequence

from ..kuka.sanitize import (MAX_BUMPS, MAX_TIME_SCALE, GripperClamp,
                             SanitizerReport, TimeScaling)
from .contract import (ACTION_DIM, ACTION_NAMES, CONTROL_HZ, GRIPPER_INDICES,
                       GRIPPER_RANGE, JOINT_INDICES, max_step_rad)

__all__ = ["SanitizerReport", "resample_path", "sanitize", "worst_velocity_ratio"]


def _finite_rows(values: list[list[float]]) -> bool:
    return all(len(r) == ACTION_DIM for r in values) and all(
        not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)
        for r in values for x in r)


def worst_velocity_ratio(values: list[list[float]], hz: float) -> tuple[float, dict]:
    """max over steps and arm joints of |step| / cap. <= 1.0 means feasible."""
    cap = max_step_rad(hz)
    worst, at = 0.0, {}
    for i in range(1, len(values)):
        for j in JOINT_INDICES:
            r = abs(values[i][j] - values[i - 1][j]) / cap
            if r > worst:
                worst, at = r, {"step": i, "joint": ACTION_NAMES[j],
                                "rad": round(abs(values[i][j] - values[i - 1][j]), 5),
                                "cap_rad": round(cap, 5)}
    return worst, at


def resample_path(values: list[list[float]], n_out: int) -> list[list[float]]:
    """Uniform, order-preserving resampling along the waypoint polyline."""
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
    """Deterministically sanitize a post-review candidate. Never mutates input."""
    rep = SanitizerReport()
    vals = [list(r) for r in rows]
    if not vals or not _finite_rows(vals):
        rep.refused = ("non-finite or misshapen array -- refused. Structurally "
                       "broken input is not sanitized, it is rejected.")
        return [list(r) for r in rows], rep

    # ---- 1. gripper clamp: the ONLY value-altering operation ----------------
    lo, hi = GRIPPER_RANGE
    for i, row in enumerate(vals):
        for g in GRIPPER_INDICES.values():
            if row[g] < lo or row[g] > hi:
                new = min(hi, max(lo, row[g]))
                rep.gripper_clamps.append(GripperClamp(i, row[g], new))
                row[g] = new
    if rep.gripper_clamps:
        rep.notes.append(
            f"clamped {len(rep.gripper_clamps)} gripper command(s) into "
            f"[{lo},{hi}]; joint angles untouched")

    # ---- 2. state bridge, only if the approach itself is infeasible ----------
    cap = max_step_rad(hz)
    bridged = False
    if state is not None and len(state) >= ACTION_DIM:
        bad = {ACTION_NAMES[j]: round(abs(vals[0][j] - state[j]), 5)
               for j in JOINT_INDICES if abs(vals[0][j] - state[j]) > cap}
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
            f"feasible is not a timing problem. Reported, not repaired.")
        rep.time_scaling = TimeScaling(applied=False, reason=rep.refused,
                                       required_scale=round(ratio, 6),
                                       worst_ratio_before=round(ratio, 6))
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
            reason=(f"velocity breach: {at.get('joint')} step {at.get('rad')} rad "
                    f"> cap {at.get('cap_rad')} rad at {hz:.0f} Hz "
                    f"(ratio {ratio:.4f})"),
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
                         "the arms' actual pose")

    rep.applied = True
    if not rep.changed:
        rep.notes.append("no change required; original array kept")
        rep.returned_unchanged = True
        return [list(r) for r in rows], rep
    return vals, rep
