"""Reading two YAM arms while they HOLD. There is no command method here.

WHAT HOLD MEANS ON THIS RIG
The KUKA cell holds by answering every 4 ms RSI frame with the measured pose.
YAM has no such handshake: i2rt runs its own CAN control thread, and
`get_yam_robot(zero_gravity_mode=False)` commands the pose the arm is at when
it connects and then keeps holding it. That is the YAM equivalent of HOLD, and
it is the only mode this module opens. It never calls `command_joint_pos`.

MOTION IS STRUCTURALLY ABSENT, NOT DISABLED
`HeldArms` exposes `read()` and `close()`. The i2rt robot objects are held
privately and no method passes a target to them, so the live loop has no route
from a model output to a motor.

THREE THINGS THAT ARE TRUE ON YAM AND WERE NOT ON THE KUKA
  1. Connecting is itself an action: the motors are ENABLED and hold with the
     driver's default gains. Stand clear, E-stop in reach.
  2. Closing disables the motors. i2rt's close() zeroes torque, so the arms go
     LIMP -- they fall under gravity unless they are resting. Start and stop in
     the configured rest pose.
  3. The grippers are on the same CAN chain as their arm (motor 0x07), so they
     hold with the arm. Linear-4310 grippers otherwise run a calibration sweep
     on connect -- that is MOTION -- so calibrated limits are REQUIRED here and
     the sweep never runs.

THE POLICY SEES THE COMMANDED GRIPPER, NOT THE ENCODER (contract.py). While
holding, the commanded opening is the hold target: the gripper reading taken
when the arm connected.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .contract import ACTION_DIM, ARMS, GRIPPER_INDICES

SCHEMA = "hybrid_rollout.robodojo.yam.robot.v1"


class RobotRefused(RuntimeError):
    """The rig cannot be opened as configured."""


@dataclass
class BimanualState:
    """One read of both arms. `measured` is the encoders; `policy_state`
    substitutes the commanded gripper opening, which is what the checkpoint
    was trained to see."""
    measured: list[float]
    policy_state: list[float]
    read_at: float
    n: int

    def to_log(self) -> dict[str, Any]:
        return {"measured": [round(v, 5) for v in self.measured],
                "policy_state": [round(v, 5) for v in self.policy_state],
                "read_at": self.read_at, "n": self.n}


class ArmReader(Protocol):
    can_move_robot: bool

    def read(self) -> BimanualState: ...

    def close(self) -> None: ...


def _default_factory(channel: str, gripper_limits: Sequence[float]):
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType
    import numpy as np
    return get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310,
                         zero_gravity_mode=False,
                         gripper_limits_override=np.asarray(gripper_limits, float))


class HeldArms:
    """Both arms, holding the pose they were connected at. Read-only."""

    can_move_robot = False

    def __init__(self, *, left_can: str, right_can: str,
                 gripper_limits: dict[str, Sequence[float]],
                 factory: Callable[[str, Sequence[float]], Any] | None = None) -> None:
        missing = [arm for arm in ARMS
                   if not gripper_limits.get(arm) or len(gripper_limits[arm]) != 2]
        if missing:
            raise RobotRefused(
                f"no calibrated [closed, open] gripper limits for {missing}. "
                f"Without them i2rt runs a calibration sweep on connect, which "
                f"moves the gripper. Calibrate once and put the limits in the "
                f"experiment config.")
        if not left_can or not right_can or left_can == right_can:
            raise RobotRefused(f"need two distinct CAN channels, got "
                               f"{left_can!r} / {right_can!r}")
        self.channels = {"left": left_can, "right": right_can}
        self.gripper_limits = {a: list(gripper_limits[a]) for a in ARMS}
        self._factory = factory or _default_factory
        self._robots: dict[str, Any] = {}
        self._hold_gripper: dict[str, float] = {}
        self._lock = threading.Lock()
        self._n = 0

    def open(self) -> "HeldArms":
        """Enables the motors and holds. An action on the robot."""
        try:
            for arm in ARMS:
                self._robots[arm] = self._factory(self.channels[arm],
                                                  self.gripper_limits[arm])
        except Exception:
            self.close()
            raise
        first = self._raw()
        for arm, g in GRIPPER_INDICES.items():
            self._hold_gripper[arm] = float(first[g])
        return self

    def _raw(self) -> list[float]:
        row: list[float] = []
        for arm in ARMS:
            q = [float(v) for v in self._robots[arm].get_joint_pos()]
            if len(q) != 7:
                raise RobotRefused(f"{arm} arm reports {len(q)} joints, expected 6 "
                                   f"+ gripper")
            row += q
        return row

    def read(self) -> BimanualState:
        with self._lock:
            if not self._robots:
                raise RobotRefused("arms are not open")
            measured = self._raw()
            self._n += 1
            n = self._n
        policy = list(measured)
        for arm, g in GRIPPER_INDICES.items():
            policy[g] = self._hold_gripper[arm]
        return BimanualState(measured, policy, time.time(), n)

    def close(self) -> None:
        """Disables the motors: the arms go limp. See module docstring."""
        with self._lock:
            robots, self._robots = self._robots, {}
        for r in robots.values():
            try:
                r.close()
            except Exception:                                  # noqa: BLE001
                pass

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "channels": self.channels,
                "gripper_limits": self.gripper_limits,
                "hold_gripper": self._hold_gripper, "reads": self._n,
                "can_move_robot": False,
                "mode": "hold (zero_gravity_mode=False); no command path"}


@dataclass
class FakeArms:
    """Test double: a fixed pose, optionally drifting. Never touches CAN."""
    pose: list[float] = field(default_factory=lambda: [0.0] * ACTION_DIM)
    drift: float = 0.0
    can_move_robot: bool = False
    reads: int = 0
    closed: bool = False

    def read(self) -> BimanualState:
        self.reads += 1
        m = [v + self.drift * self.reads for v in self.pose]
        return BimanualState(m, list(m), time.time(), self.reads)

    def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------ execution
def open_for_motion(*, left_can: str, right_can: str,
                    gripper_limits: dict[str, Sequence[float]],
                    factory: Callable[[str, Sequence[float]], Any] | None = None
                    ) -> dict[str, Any]:
    """Open both arms for `motion.MotionOwner`, and ONLY for it.

    Same preconditions as HeldArms (two channels, calibrated grippers, so no
    sweep on connect). The robots come up holding their current pose; the
    returned objects are handed straight to the motion owner, which is the
    single caller of command_joint_pos in this package.
    """
    HeldArms(left_can=left_can, right_can=right_can, gripper_limits=gripper_limits)
    make = factory or _default_factory
    robots: dict[str, Any] = {}
    try:
        for arm, ch in (("left", left_can), ("right", right_can)):
            robots[arm] = make(ch, gripper_limits[arm])
    except Exception:
        for r in robots.values():
            try:
                r.close()
            except Exception:                                  # noqa: BLE001
                pass
        raise
    return robots


def calibrate_gripper(channel: str) -> list[float]:
    """Run i2rt's own gripper calibration on one arm and return [closed, open].

    MOTION: with no limits supplied, i2rt sweeps the gripper to its hard stops
    on connect. The arm holds its pose throughout; the motors are disabled
    again when this returns, so the arm goes limp -- support it or rest it.
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType
    robot = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310,
                          zero_gravity_mode=False)
    try:
        limits = robot.get_robot_info().get("gripper_limits")
        if limits is None or len(limits) != 2:
            raise RobotRefused(f"{channel}: i2rt reported no gripper limits")
        return [float(v) for v in limits]
    finally:
        robot.close()
