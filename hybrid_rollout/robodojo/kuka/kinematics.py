"""Single-arm FK preview for the KUKA, reusing upstream ArmFK.

Upstream `robodojo_server/kinematics.py` already resolves a URDF chain and does
robot-only FK; there is no reason to re-derive that, so `ArmFK` is used directly
and only the dual-arm wrapper is replaced.

WHAT AN FK PREVIEW IS, AND IS NOT
Upstream states it plainly in SKILL.md and the wording is kept: "FK is not a
simulation of contact, grasping, objects or future success." It maps commanded
joint targets to where the flange would be if they were executed. It predicts
nothing about the world.

TWO UNVERIFIED THINGS, BOTH GATING eef
  1. the custom $TOOL/TCP transform is unknown, so flange != tool tip;
  2. the wrist sign conventions are unconfirmed -- a single logged pose cannot
     resolve them (8 of 64 sign combinations fitted one sample within 1 mm).
Consequently `target()` REFUSES, and `eef` never reaches the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .contract import (ARM_DIM, EEF_EXECUTION_ENABLED, EEF_WITHHELD_REASON,
                       ROBOT_MODEL_SOURCE, eef_execution_gate)

# URDF joint names differ from the dataset's action names (A1..A6).
URDF_JOINT_NAMES = tuple(f"joint_{i + 1}" for i in range(ARM_DIM))
DEFAULT_URDF = Path(__file__).parent / "robot_model" / "lbr_iisy11_r1300.urdf"
BASE_LINK, TIP_LINK = "base_link", "flange"

INTERPRETATION = "kinematic_targets_not_object_future"


@dataclass
class FKPreview:
    available: bool
    reason: str = ""
    frame: str | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    interpretation: str = INTERPRETATION
    source: str | None = None
    tool_tip_accurate: bool = False        # never true without a verified $TOOL

    def to_log(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason,
                "frame": self.frame, "interpretation": self.interpretation,
                "source": self.source, "n_poses": len(self.trajectory),
                "tool_tip_accurate": self.tool_tip_accurate,
                "caveat": ("flange-frame only; the custom $TOOL/TCP transform is "
                           "unverified, so these are not tool-tip positions"),
                "trajectory": self.trajectory}


class FKProvider(Protocol):
    available: bool

    def preview(self, joint_rows_deg: Sequence[Sequence[float]]) -> FKPreview: ...


class UnavailableFK:
    """Default. Reports unavailability instead of inventing geometry."""

    available = False

    def __init__(self, reason: str = "") -> None:
        self.reason = reason or "no FK provider configured"

    def preview(self, joint_rows_deg) -> FKPreview:       # noqa: D102
        return FKPreview(available=False, reason=self.reason)

    def target(self, *_a, **_k):                          # noqa: D102
        raise RuntimeError(EEF_WITHHELD_REASON)


class KukaArmFK:
    """Wraps upstream ArmFK for one 6-DoF arm. numpy/scipy only, no robot I/O."""

    available = True

    def __init__(self, urdf: str | Path = DEFAULT_URDF,
                 joint_names: Sequence[str] = URDF_JOINT_NAMES,
                 base: str = BASE_LINK, tip: str = TIP_LINK) -> None:
        from ..robodojo_server.kinematics import ArmFK     # upstream, unchanged
        self.urdf = str(urdf)
        self.fk = ArmFK(self.urdf, list(joint_names), base=base, tip=tip)
        self.frame = f"{base}->{tip}"

    def preview(self, joint_rows_deg) -> FKPreview:
        import math

        import numpy as np
        rows = []
        for i, q_deg in enumerate(joint_rows_deg):
            q = [math.radians(float(v)) for v in list(q_deg)[:ARM_DIM]]
            m = self.fk.matrix(np.asarray(q, float))
            rows.append({"step": i,
                         "position": [float(v) for v in m[:3, 3]],
                         "joints_deg": [round(float(v), 4) for v in list(q_deg)[:ARM_DIM]]})
        return FKPreview(available=True, frame=self.frame, trajectory=rows,
                         source=ROBOT_MODEL_SOURCE)

    def target(self, *_a, **_k):
        """IK. Refused: see module docstring."""
        raise RuntimeError(EEF_WITHHELD_REASON)


def make_fk(urdf: str | Path = DEFAULT_URDF) -> FKProvider:
    """Best available provider. Missing numpy/scipy is NOT an error -- FK is a
    diagnostic, and the loop must run on a Jetson that may not carry scipy."""
    try:
        return KukaArmFK(urdf)
    except Exception as exc:                               # noqa: BLE001
        return UnavailableFK(f"{type(exc).__name__}: {exc}"[:200])


def eef_enabled() -> tuple[bool, str]:
    allowed, _missing, reason = eef_execution_gate()
    return allowed, reason
