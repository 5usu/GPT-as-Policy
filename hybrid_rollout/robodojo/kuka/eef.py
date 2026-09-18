"""Cartesian (eef) support for Astra-direct, gated on operator-supplied facts.

WHY THIS MODULE IS CAUTIOUS
`eef` is the mode where Astra emits a tool pose instead of joint numbers, and it
is the ONLY mode upstream's Direct evaluation allows (gpt_only_client.py refuses
joint actions outright). Enabling it therefore unblocks "Astra does it itself" --
and it is also the mode with the least local checking available, because we do
not resolve a Cartesian target into joint angles. Something else does.

WHO DOES THE INVERSE KINEMATICS MATTERS
The KUKA controller can resolve Cartesian corrections itself, via RSI's <RKorr>
in a Cartesian-configured RSI context. If that is how the cell is set up, we do
not need our own IK -- but we then CANNOT check the resulting joint angles
before they exist, because we never compute them. Our defence is therefore
entirely upstream of the controller:

  1. the target must be within a bounded distance of the MEASURED pose (RIst),
     so a single step can only ever be a small correction;
  2. the target must lie inside an operator-supplied workspace volume;
  3. a collision model must exist;
  4. the RSI configuration must actually be Cartesian-capable.

EVIDENCE NOTE, recorded honestly: the RSI .src files and Python in
KUKA/teleoperation are JOINT-SPACE ONLY (AIPos in, AK out). Nothing there
references TOOL, BASE, RIst or RKorr. So a Cartesian path is NOT demonstrated by
the existing deployment and requires a different RSI configuration plus the
$TOOL value read off the controller. `CartesianCapability` records which of
those an operator has actually attested, and nothing is assumed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.eef.v1"

#: Upstream bounds for one takeover step (SKILL.md): 5 cm and 0.35 rad per arm.
MAX_POSITION_DELTA_M = 0.05
MAX_ROTATION_DELTA_RAD = 0.35


@dataclass
class CartesianCapability:
    """What the operator has attested about this cell. Nothing defaults to true."""
    tcp_transform: Any = None            # $TOOL read off the controller
    workspace: Any = None                # allowed Cartesian volume
    collision_model: Any = None
    rsi_cartesian_configured: bool = False   # RSI context emits/accepts RKorr
    rist_readback_verified: bool = False     # RIst confirmed against known poses
    resolver: str = "controller"             # who turns Cartesian into joints

    def missing(self) -> list[str]:
        out = []
        if self.tcp_transform in (None, "", [], {}):
            out.append("tcp_transform ($TOOL read off the controller)")
        if self.workspace in (None, "", [], {}):
            out.append("workspace (allowed Cartesian volume)")
        if self.collision_model in (None, "", [], {}):
            out.append("collision_model")
        if not self.rsi_cartesian_configured:
            out.append("rsi_cartesian_configured (the deployed .src is joint-only)")
        if not self.rist_readback_verified:
            out.append("rist_readback_verified (RIst checked against known poses)")
        return out

    def allowed(self) -> tuple[bool, list[str]]:
        m = self.missing()
        return (not m), m

    def to_log(self) -> dict[str, Any]:
        ok, missing = self.allowed()
        return {"schema": SCHEMA, "eef_execution_allowed": ok, "missing": missing,
                "resolver": self.resolver,
                "rsi_cartesian_configured": self.rsi_cartesian_configured,
                "rist_readback_verified": self.rist_readback_verified,
                "note": ("Cartesian targets are resolved by the CONTROLLER, so the "
                         "resulting joint angles cannot be checked here before "
                         "they exist. Bounding the step against the measured pose "
                         "is the defence.")}


@dataclass
class EefViolation:
    code: str
    detail: str
    magnitude: float | None = None
    limit: float | None = None

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


def _quat_norm(q: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in q))


def quat_angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in radians between two wxyz quaternions."""
    na, nb = _quat_norm(a), _quat_norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero-norm quaternion")
    d = sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)
    return 2.0 * math.acos(min(1.0, abs(d)))


def inside_workspace(position: Sequence[float], workspace: Any) -> bool:
    """Axis-aligned box check. An unparseable workspace is NOT inside."""
    try:
        lo = [float(v) for v in workspace["min"]]
        hi = [float(v) for v in workspace["max"]]
    except Exception:
        return False
    return all(lo[i] <= float(position[i]) <= hi[i] for i in range(3))


def validate_target(target: dict[str, Any], *, measured: dict[str, Any],
                    capability: CartesianCapability,
                    max_position_delta_m: float = MAX_POSITION_DELTA_M,
                    max_rotation_delta_rad: float = MAX_ROTATION_DELTA_RAD
                    ) -> list[EefViolation]:
    """Bound one Cartesian target against the measured pose. FAIL-CLOSED."""
    v: list[EefViolation] = []
    ok, missing = capability.allowed()
    if not ok:
        v.append(EefViolation("eef_not_capable",
                              "cell has not attested: " + ", ".join(missing)))
        return v                      # do not pretend to check further

    pos = target.get("position")
    quat = target.get("quaternion_wxyz")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        v.append(EefViolation("bad_position", "position must be 3 numbers"))
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        v.append(EefViolation("bad_quaternion", "quaternion_wxyz must be 4 numbers"))
    if v:
        return v
    for name, seq in (("position", pos), ("quaternion", quat)):
        for x in seq:
            if isinstance(x, bool) or not isinstance(x, (int, float)) \
                    or not math.isfinite(x):
                v.append(EefViolation("non_finite", f"{name} contains {x!r}"))
                return v
    if _quat_norm(quat) < 1e-6:
        v.append(EefViolation("degenerate_quaternion", "zero-norm orientation"))
        return v

    mpos = measured.get("position")
    mquat = measured.get("quaternion_wxyz")
    if not mpos or not mquat:
        v.append(EefViolation("no_measured_pose",
                              "no measured RIst pose supplied; refusing to bound "
                              "a target against an unknown current pose"))
        return v

    d = math.dist([float(x) for x in pos], [float(x) for x in mpos])
    if d > max_position_delta_m:
        v.append(EefViolation("position_delta", f"{d:.4f} m from measured pose",
                              round(d, 6), max_position_delta_m))
    try:
        ang = quat_angle_between(quat, mquat)
    except ValueError as exc:
        v.append(EefViolation("degenerate_quaternion", str(exc)))
        return v
    if ang > max_rotation_delta_rad:
        v.append(EefViolation("rotation_delta", f"{ang:.4f} rad from measured pose",
                              round(ang, 6), max_rotation_delta_rad))
    if not inside_workspace(pos, capability.workspace):
        v.append(EefViolation("outside_workspace",
                              f"position {[round(float(x), 4) for x in pos]} "
                              f"outside the configured volume"))
    return v


def eef_eligible(violations: list[EefViolation]) -> tuple[bool, list[str]]:
    """FAIL-CLOSED: any violation blocks."""
    return (not violations), sorted({x.code for x in violations})
