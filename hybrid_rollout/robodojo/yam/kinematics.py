"""Per-arm FK preview for bimanual YAM, reusing upstream ArmFK.

Each arm is previewed in ITS OWN base frame. Where the two bases sit relative
to each other (and to the cameras) has not been calibrated on this rig, so the
two trajectories are not in a common frame and must not be compared as if they
were. The tip is the URDF `gripper` link, which i2rt documents as the
end-effector MOUNT frame, not a finger tip.

As upstream says, and as the KUKA port kept: FK is not a simulation of
contact, grasping, objects or future success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..kuka.kinematics import INTERPRETATION, UnavailableFK
from .contract import ARM_JOINT_INDICES, ARMS, EEF_WITHHELD_REASON, ROBOT_MODEL_SOURCE

DEFAULT_URDF = Path(__file__).parent / "robot_model" / "yam.urdf"
URDF_JOINT_NAMES = tuple(f"joint{i + 1}" for i in range(6))
BASE_LINK, TIP_LINK = "base", "gripper"

__all__ = ["BimanualFK", "BimanualFKPreview", "UnavailableFK", "make_fk"]


@dataclass
class BimanualFKPreview:
    available: bool
    reason: str = ""
    frame: str | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    interpretation: str = INTERPRETATION
    source: str | None = None
    tool_tip_accurate: bool = False
    common_frame: bool = False          # the two arms are NOT in one frame

    def to_log(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason,
                "frame": self.frame, "interpretation": self.interpretation,
                "source": self.source, "n_poses": len(self.trajectory),
                "tool_tip_accurate": self.tool_tip_accurate,
                "common_frame": self.common_frame,
                "caveat": ("per-arm base frame, gripper MOUNT link; the arm "
                           "bases are not calibrated into one frame and these "
                           "are not finger-tip positions"),
                "trajectory": self.trajectory}


class BimanualFK:
    """Two upstream ArmFK chains over one URDF. numpy/scipy only, no robot I/O."""

    available = True

    def __init__(self, urdf: str | Path = DEFAULT_URDF) -> None:
        from ..robodojo_server.kinematics import ArmFK     # upstream, unchanged
        self.urdf = str(urdf)
        self.fk = ArmFK(self.urdf, list(URDF_JOINT_NAMES), base=BASE_LINK, tip=TIP_LINK)
        self.frame = f"{{left,right}}/{BASE_LINK}->{TIP_LINK}"

    def preview(self, rows: Sequence[Sequence[float]]) -> BimanualFKPreview:
        import numpy as np
        out = []
        for i, row in enumerate(rows):
            r = list(row)
            step: dict[str, Any] = {"step": i}
            for arm in ARMS:
                q = [float(r[j]) for j in ARM_JOINT_INDICES[arm]]
                m = self.fk.matrix(np.asarray(q, float))
                step[arm] = [float(v) for v in m[:3, 3]]
            out.append(step)
        return BimanualFKPreview(available=True, frame=self.frame,
                                 trajectory=out, source=ROBOT_MODEL_SOURCE)

    def target(self, *_a, **_k):
        """IK. Refused: see contract.EEF_WITHHELD_REASON."""
        raise RuntimeError(EEF_WITHHELD_REASON)


def make_fk(urdf: str | Path = DEFAULT_URDF):
    """Best available provider. Missing numpy/scipy is not an error."""
    try:
        return BimanualFK(urdf)
    except Exception as exc:                               # noqa: BLE001
        return UnavailableFK(f"{type(exc).__name__}: {exc}"[:200])
