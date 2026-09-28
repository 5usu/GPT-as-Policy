"""The one thing in this package that commands the YAM arms.

A jerk-limited Ruckig reference at 100 Hz, fed with 30 Hz policy targets --
the same structure as the published YAM pi0.5 controller (Agents2AgentsAI/
vla-edge examples/bimanual-yam/pi05_motion.py): the reference advances from
its OWN previous state, never from measured-position error, so a slow model
cycle or a late camera frame cannot produce a jump. Between policy cycles the
reference simply comes to rest at the last goal and holds.

WHAT STOPS IT
  - fault: measured arm joints deviate from the reference by more than the
    configured tolerance for DEVIATION_TICKS consecutive ticks (0.1 s). The
    reference brakes to zero velocity and holds; no further goal is accepted.
  - brake(): operator stop. Same brake, same hold.
  - a Ruckig error or a non-finite reading: fault.
The hardware E-stop is authoritative and outside software.

WHAT IT REFUSES
  - to start without ruckig (no substitute profile, as on the KUKA branch);
  - a goal whose arm joints leave the enforced limits, or that is not 14
    finite values. Grippers are clamped to [0, 1] by Ruckig's target, as in
    the reference controller.

Nothing here decides WHAT to command. `execute.py` validates and sanitizes
every chunk first; this module only turns an approved goal into bounded motion.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .contract import (ACTION_DIM, ARM_JOINT_INDICES, ARMS, GRIPPER_INDICES,
                       JOINT_INDICES, arm_joint_limit)

SCHEMA = "yam.motion.v1"
DEVIATION_TICKS = 10


class MotionRefused(RuntimeError):
    """Motion could not be started or a goal was refused."""


@dataclass(frozen=True)
class MotionLimits:
    """Per-joint limits for the reference. Arm values are rad, rad/s, ..."""
    hz: float = 100.0
    velocity: float = 0.5
    acceleration: float = 1.5
    jerk: float = 10.0
    gripper_velocity: float = 2.0
    gripper_acceleration: float = 10.0
    gripper_jerk: float = 100.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise MotionRefused("every motion limit must be finite and positive")
        if self.velocity > 2.2:
            raise MotionRefused(
                f"arm velocity {self.velocity} rad/s exceeds 2.2 rad/s, the "
                f"ceiling the published YAM pi0.5 controller runs at")

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


def _vector(arm: float, grip: float) -> list[float]:
    v = [arm] * ACTION_DIM
    for g in GRIPPER_INDICES.values():
        v[g] = grip
    return v


class Reference:
    """Ruckig state for 14 dims. Pure computation; touches no hardware."""

    def __init__(self, q: Sequence[float], limits: MotionLimits) -> None:
        try:
            from ruckig import InputParameter, OutputParameter, Ruckig, Synchronization
        except ImportError:
            raise MotionRefused("ruckig is not installed; no substitute motion "
                                "profile is used") from None
        q = [float(v) for v in q]
        if len(q) != ACTION_DIM or not all(math.isfinite(v) for v in q):
            raise MotionRefused("invalid initial reference")
        self.limits = limits
        self.dt = 1.0 / limits.hz
        self.otg = Ruckig(ACTION_DIM, self.dt)
        self.inp = InputParameter(ACTION_DIM)
        self.out = OutputParameter(ACTION_DIM)
        self.inp.current_position = q
        self.inp.current_velocity = [0.0] * ACTION_DIM
        self.inp.current_acceleration = [0.0] * ACTION_DIM
        self.inp.target_position = list(q)
        self.inp.target_velocity = [0.0] * ACTION_DIM
        self.inp.target_acceleration = [0.0] * ACTION_DIM
        self.inp.synchronization = Synchronization.Time
        self.inp.max_velocity = _vector(limits.velocity, limits.gripper_velocity)
        self.inp.max_acceleration = _vector(limits.acceleration, limits.gripper_acceleration)
        self.inp.max_jerk = _vector(limits.jerk, limits.gripper_jerk)
        self.position = list(q)
        self.velocity = [0.0] * ACTION_DIM

    def target(self, q: Sequence[float]) -> None:
        q = [float(v) for v in q]
        for g in GRIPPER_INDICES.values():
            q[g] = min(1.0, max(0.0, q[g]))
        self.inp.target_position = q

    def brake(self) -> None:
        from ruckig import ControlInterface, Synchronization
        self.inp.control_interface = ControlInterface.Velocity
        self.inp.synchronization = Synchronization.No
        self.inp.target_velocity = [0.0] * ACTION_DIM
        self.inp.target_acceleration = [0.0] * ACTION_DIM

    def tick(self) -> tuple[list[float], bool]:
        from ruckig import Result
        res = self.otg.update(self.inp, self.out)
        if res not in (Result.Working, Result.Finished):
            raise MotionRefused(f"ruckig failed: {res}")
        self.position = list(self.out.new_position)
        self.velocity = list(self.out.new_velocity)
        self.out.pass_to_input(self.inp)
        return list(self.position), res == Result.Finished


def goal_problem(q: Sequence[float]) -> str | None:
    q = list(q)
    if len(q) != ACTION_DIM or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in q):
        return f"goal must be {ACTION_DIM} finite values"
    for j in JOINT_INDICES:
        lo, hi = arm_joint_limit(j)
        if not lo <= q[j] <= hi:
            return f"goal joint {j} = {q[j]:.4f} outside [{lo:.4f}, {hi:.4f}]"
    return None


class MotionOwner:
    """Owns the robot objects. The only caller of command_joint_pos."""

    can_move_robot = True

    def __init__(self, robots: dict[str, Any], limits: MotionLimits, *,
                 tolerance_rad: float,
                 on_fault: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if set(robots) != set(ARMS):
            raise MotionRefused(f"need robots for {ARMS}, got {sorted(robots)}")
        if not (isinstance(tolerance_rad, (int, float)) and 0 < tolerance_rad < 1.0):
            raise MotionRefused("commanded_observed_tolerance_rad must be in (0, 1) rad")
        self.robots = robots
        self.limits = limits
        self.tolerance = float(tolerance_rad)
        self.on_fault = on_fault
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ref: Reference | None = None
        self.goal: list[float] | None = None
        self.measured: list[float] | None = None
        self.done = True
        self.fault: str | None = None
        self.braking = False
        self.ticks = 0
        self.overruns = 0
        self.max_deviation = 0.0
        self._over = 0

    # ------------------------------------------------------------ hardware
    def read(self) -> list[float]:
        row: list[float] = []
        for arm in ARMS:
            q = [float(v) for v in self.robots[arm].get_joint_pos()]
            if len(q) != 7 or not all(math.isfinite(v) for v in q):
                raise MotionRefused(f"{arm} returned invalid joint state {q}")
            row += q
        return row

    def _command(self, q: Sequence[float]) -> None:
        import numpy as np
        for arm in ARMS:
            idx = list(ARM_JOINT_INDICES[arm]) + [GRIPPER_INDICES[arm]]
            self.robots[arm].command_joint_pos(np.asarray([q[i] for i in idx], float))

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "MotionOwner":
        q0 = self.read()
        self.measured = q0
        self.ref = Reference(q0, self.limits)
        self.goal = list(q0)
        self._thread = threading.Thread(target=self._loop, name="yam-motion", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        dt = self.ref.dt
        last = None
        while not self._stop.is_set():
            if last is not None:
                wait = last + dt - self._clock()
                if wait > 0:
                    self._sleep(wait)
            now = self._clock()
            if last is not None and now - last > 5 * dt:
                self.overruns += 1
            last = now
            try:
                with self._lock:
                    q, done = self.ref.tick()
                    self.done = done
                self._command(q)
                m = self.read()
                self.measured = m
                self.ticks += 1
                dev = max(abs(m[j] - q[j]) for j in JOINT_INDICES)
                self.max_deviation = max(self.max_deviation, dev)
                self._over = self._over + 1 if dev > self.tolerance else 0
                if self._over >= DEVIATION_TICKS and self.fault is None:
                    self._trip(f"measured-vs-reference deviation {dev:.4f} rad > "
                               f"{self.tolerance} rad for {self._over} ticks")
            except Exception as exc:                           # noqa: BLE001
                if self.fault is None:
                    self._trip(f"{type(exc).__name__}: {exc}"[:200])
                self._sleep(dt)

    def _trip(self, why: str) -> None:
        self.fault = why
        self.brake()
        if self.on_fault:
            try:
                self.on_fault(why)
            except Exception:                                  # noqa: BLE001
                pass

    # ------------------------------------------------------------ commands
    def set_goal(self, q: Sequence[float]) -> None:
        if self.fault:
            raise MotionRefused(f"motion faulted: {self.fault}")
        if self.braking:
            raise MotionRefused("motion is braked; no new goal is accepted")
        bad = goal_problem(q)
        if bad:
            raise MotionRefused(bad)
        with self._lock:
            self.ref.target(q)
            self.goal = [float(v) for v in q]
            self.done = False

    def brake(self) -> None:
        with self._lock:
            self.braking = True
            if self.ref is not None:
                self.ref.brake()

    def wait_settled(self, timeout_s: float) -> bool:
        t0 = self._clock()
        while self._clock() - t0 < timeout_s:
            if self.done or self.fault:
                return self.done and not self.fault
            self._sleep(0.01)
        return False

    def park(self, target: Sequence[float], limits: MotionLimits,
             timeout_s: float = 60.0, *, after_fault: bool = False) -> bool:
        """Move slowly to `target` (the rest pose) after a stop. Blocking.

        Refused after a fault unless a human said so: a deviation fault can
        mean the arm is in contact with something, and moving it again is a
        decision for the person looking at it.
        """
        if self.fault and not after_fault:
            raise MotionRefused(f"not parking after a fault ({self.fault}) "
                                f"without an explicit operator decision")
        bad = goal_problem(target)
        if bad:
            raise MotionRefused(f"rest pose refused: {bad}")
        with self._lock:
            self.ref = Reference(self.ref.position if self.ref else self.read(), limits)
            self.ref.target(target)
            self.braking = False
            self.fault = None
            self._over = 0
            self.done = False
        return self.wait_settled(timeout_s)

    def retune(self, limits: MotionLimits) -> None:
        """Swap limits once the reference is at rest (after a park)."""
        if not self.done:
            raise MotionRefused("cannot change limits while moving")
        with self._lock:
            self.ref = Reference(self.ref.position, limits)
            self.limits = limits
            self.done = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def status(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "ticks": self.ticks, "overruns": self.overruns,
                "fault": self.fault, "braking": self.braking, "done": self.done,
                "max_deviation_rad": round(self.max_deviation, 5),
                "tolerance_rad": self.tolerance, "limits": self.limits.to_log()}
