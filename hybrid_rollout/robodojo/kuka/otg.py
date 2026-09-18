"""Online trajectory generation: the production Ruckig contract, 6 joints, 4 ms.

Replaces the linear bridge, which was velocity-DISCONTINUOUS at every segment
boundary and therefore not what the cell runs. The deployed Jetson builds
`Ruckig(NUM_JOINTS, RSI_CYCLE_TIME)` with per-joint velocity, acceleration and
jerk limits; this module honours that same contract.

LIMITS COME FROM VALIDATED DEPLOYMENT CONFIGURATION ONLY.
There is no default limit set here and there will not be one. A plausible number
looks like knowledge, and a jerk limit that is merely plausible produces motion
that is merely plausibly safe. `JointLimits.from_config` accepts only a complete
set; anything missing REFUSES ARMING and names what is absent.

For reference when the deployment engineer fills the config, the deployed
`udp_teleoperate.py` currently uses:

    MAX_VELOCITY     [85, 40, 125, 125, 125, 320]        deg/s
    MAX_ACCELERATION [800, 250, 500, 1200, 1200, 2500]   deg/s^2
    MAX_JERK         [12000, 12000, 15000, 20000, 20000, 40000] deg/s^3

Those are recorded as a REFERENCE, not a default: they belong to the teleop
tuning and must be confirmed for this task before use. Nothing reads them
automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .contract import ARM_DIM, POSITION_LIMIT_DEG

SCHEMA = "hybrid_rollout.robodojo.kuka.otg.v1"

CYCLE_TIME_S = 0.004
NUM_JOINTS = ARM_DIM

#: Reference only. NEVER used as a fallback; see module docstring.
DEPLOYED_TELEOP_REFERENCE = {
    "max_velocity_deg_s": [85.0, 40.0, 125.0, 125.0, 125.0, 320.0],
    "max_acceleration_deg_s2": [800.0, 250.0, 500.0, 1200.0, 1200.0, 2500.0],
    "max_jerk_deg_s3": [12000.0, 12000.0, 15000.0, 20000.0, 20000.0, 40000.0],
    "source": "KUKA/teleoperation/udp_teleoperate.py:135-141 (teleop tuning)",
    "is_a_default": False,
    "must_be_confirmed_for_this_task": True,
}


class LimitsMissing(Exception):
    """Raised instead of substituting a plausible limit."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "joint limits absent from validated deployment configuration: "
            + ", ".join(self.missing)
            + ". Refusing to arm. These are not defaulted: a plausible jerk "
              "limit produces plausibly-safe motion, which is not the same thing.")


@dataclass(frozen=True)
class JointLimits:
    max_velocity: tuple[float, ...]
    max_acceleration: tuple[float, ...]
    max_jerk: tuple[float, ...]
    source: str = "deployment_config"

    @staticmethod
    def _need(cfg: dict[str, Any], key: str, missing: list[str]):
        v = cfg.get(key)
        if v in (None, "", [], {}):
            missing.append(key)
            return None
        seq = list(v)
        if len(seq) != NUM_JOINTS:
            missing.append(f"{key} (need {NUM_JOINTS} values, got {len(seq)})")
            return None
        for i, x in enumerate(seq):
            if not isinstance(x, (int, float)) or isinstance(x, bool) or x <= 0:
                missing.append(f"{key}[{i}] must be a positive number, got {x!r}")
                return None
        return tuple(float(x) for x in seq)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "JointLimits":
        missing: list[str] = []
        vel = cls._need(cfg, "max_velocity_deg_s", missing)
        acc = cls._need(cfg, "max_acceleration_deg_s2", missing)
        jrk = cls._need(cfg, "max_jerk_deg_s3", missing)
        if missing:
            raise LimitsMissing(missing)
        return cls(vel, acc, jrk, source=str(cfg.get("source", "deployment_config")))

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["cycle_time_s"] = CYCLE_TIME_S
        d["num_joints"] = NUM_JOINTS
        return d


@dataclass
class OtgResult:
    cycles: list[list[float]] = field(default_factory=list)
    velocities: list[list[float]] = field(default_factory=list)
    accelerations: list[list[float]] = field(default_factory=list)
    finished: bool = False
    n_cycles: int = 0
    duration_s: float = 0.0
    backend: str = ""
    error: str | None = None

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "backend": self.backend,
                "n_cycles": self.n_cycles, "duration_s": round(self.duration_s, 6),
                "finished": self.finished, "error": self.error,
                "first_cycle": [round(v, 4) for v in (self.cycles[0] if self.cycles else [])],
                "last_cycle": [round(v, 4) for v in (self.cycles[-1] if self.cycles else [])]}


class RuckigOTG:
    """The production contract: Ruckig(6, 0.004), jerk-limited, per-joint limits.

    `deployable` is True only when the real `ruckig` package is importable AND a
    complete validated limit set was supplied. Either missing is a refusal, not
    a downgrade.
    """

    name = "ruckig"

    def __init__(self, limits: JointLimits, *, cycle_time: float = CYCLE_TIME_S,
                 max_cycles: int = 5000) -> None:
        self.limits = limits
        self.cycle_time = cycle_time
        self.max_cycles = max_cycles
        self._available, self._why = self._probe()

    @staticmethod
    def _probe() -> tuple[bool, str]:
        try:
            import ruckig  # noqa: F401
        except ImportError as exc:
            return False, (f"the `ruckig` package is not importable ({exc}). This "
                           f"build does not substitute another profile for it.")
        return True, "ruckig available"

    @property
    def deployable(self) -> bool:
        return self._available

    @property
    def reason_not_deployable(self) -> str:
        return "" if self._available else self._why

    def generate(self, start: Sequence[float], target: Sequence[float], *,
                 start_velocity: Sequence[float] | None = None,
                 start_acceleration: Sequence[float] | None = None,
                 target_velocity: Sequence[float] | None = None) -> OtgResult:
        """Jerk-limited profile from (pos, vel, acc) to the target, at 4 ms."""
        if not self._available:
            return OtgResult(backend=self.name, error=self._why)
        from ruckig import InputParameter, OutputParameter, Result, Ruckig

        otg = Ruckig(NUM_JOINTS, self.cycle_time)
        inp = InputParameter(NUM_JOINTS)
        inp.current_position = [float(v) for v in list(start)[:NUM_JOINTS]]
        inp.current_velocity = [float(v) for v in (start_velocity or [0.0] * NUM_JOINTS)]
        inp.current_acceleration = [float(v) for v in
                                    (start_acceleration or [0.0] * NUM_JOINTS)]
        inp.target_position = [float(v) for v in list(target)[:NUM_JOINTS]]
        inp.target_velocity = [float(v) for v in (target_velocity or [0.0] * NUM_JOINTS)]
        inp.target_acceleration = [0.0] * NUM_JOINTS
        inp.max_velocity = list(self.limits.max_velocity)
        inp.max_acceleration = list(self.limits.max_acceleration)
        inp.max_jerk = list(self.limits.max_jerk)

        out = OutputParameter(NUM_JOINTS)
        res = OtgResult(backend=self.name)
        for _ in range(self.max_cycles):
            r = otg.update(inp, out)
            if r == Result.Error or int(r) < 0:
                res.error = f"ruckig returned {r}"
                return res
            res.cycles.append([float(v) for v in out.new_position])
            res.velocities.append([float(v) for v in out.new_velocity])
            res.accelerations.append([float(v) for v in out.new_acceleration])
            if r == Result.Finished:
                res.finished = True
                break
            out.pass_to_input(inp)
        res.n_cycles = len(res.cycles)
        res.duration_s = res.n_cycles * self.cycle_time
        if not res.finished and res.error is None:
            res.error = (f"profile did not converge within {self.max_cycles} cycles "
                         f"({res.duration_s:.2f} s)")
        return res

    # Callable form, so it drops into RSIGateway's interpolator slot.
    def __call__(self, start: Sequence[float], target: Sequence[float],
                 cycles: int | None = None) -> list[list[float]]:
        r = self.generate(start, target)
        if r.error:
            raise RuntimeError(f"ruckig OTG failed: {r.error}")
        return r.cycles


def check_profile(result: OtgResult, limits: JointLimits, *,
                  cycle_time: float = CYCLE_TIME_S,
                  start: Sequence[float] | None = None) -> list[str]:
    """Deterministic continuity and limit compliance over a generated profile."""
    problems: list[str] = []
    if result.error:
        return [result.error]
    if not result.cycles:
        return ["empty profile"]

    prev = list(start)[:NUM_JOINTS] if start is not None else result.cycles[0]
    for i, row in enumerate(result.cycles):
        for j in range(NUM_JOINTS):
            lo, hi = POSITION_LIMIT_DEG[j]
            if not lo <= row[j] <= hi:
                problems.append(f"cycle {i} J{j+1}: position {row[j]:.3f} outside "
                                f"[{lo}, {hi}]")
            step = abs(row[j] - prev[j])
            cap = limits.max_velocity[j] * cycle_time
            if step > cap * 1.001 + 1e-9:
                problems.append(f"cycle {i} J{j+1}: step {step:.5f} deg exceeds "
                                f"v_max*dt {cap:.5f}")
        prev = row

    for i, v in enumerate(result.velocities):
        for j in range(NUM_JOINTS):
            if abs(v[j]) > limits.max_velocity[j] * 1.001 + 1e-9:
                problems.append(f"cycle {i} J{j+1}: |v| {abs(v[j]):.3f} > "
                                f"{limits.max_velocity[j]}")
    for i, a in enumerate(result.accelerations):
        for j in range(NUM_JOINTS):
            if abs(a[j]) > limits.max_acceleration[j] * 1.001 + 1e-9:
                problems.append(f"cycle {i} J{j+1}: |a| {abs(a[j]):.3f} > "
                                f"{limits.max_acceleration[j]}")
    # Jerk is the derivative of the acceleration the generator reports.
    for i in range(1, len(result.accelerations)):
        for j in range(NUM_JOINTS):
            jerk = abs(result.accelerations[i][j] - result.accelerations[i-1][j]) / cycle_time
            if jerk > limits.max_jerk[j] * 1.05 + 1e-6:
                problems.append(f"cycle {i} J{j+1}: |jerk| {jerk:.1f} > "
                                f"{limits.max_jerk[j]}")
    return problems
