"""Commanded-versus-measured monitoring, every 4 ms cycle.

WHAT THIS CATCHES THAT NOTHING ELSE DOES
Validation checks that a command is sensible. This checks that the arm actually
did it. Those diverge when something physical is wrong -- the gripper snagged,
the door bound, a joint is fighting a load -- and the command stream looks
perfectly healthy throughout.

WHY DEBOUNCE, AND WHY IT IS NOT JUST NOISE FILTERING
A single sample over tolerance is usually lag: the controller is a cycle behind
a fast move. Latching FAULT on one sample would stop constantly. But treating
every deviation as noise means a real collision reads as a run of tolerable
samples. So: one isolated breach enters HOLD (stop moving, keep answering),
while a run of them, or a single severe one, latches FAULT. Both thresholds come
from configuration and neither is defaulted.

EVERY DECISION IS RECORDED WITH ITS INPUTS -- commanded, interpolated, measured,
residual, which limit it hit and why -- because "it stopped and we don't know
why" is the failure mode that ends an experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from .contract import ACTION_NAMES, ARM_DIM

SCHEMA = "hybrid_rollout.robodojo.kuka.monitor.v1"


class Verdict(str, Enum):
    OK = "ok"
    HOLD = "hold"            # isolated breach: stop moving, keep replying
    FAULT = "fault"          # persistent or severe: latched, needs a human


class ToleranceMissing(Exception):
    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "commanded-vs-measured tolerances absent from deployment "
            "configuration: " + ", ".join(self.missing) + ". Refusing to arm. "
            "Without them a deviation cannot be judged, and an unjudgeable "
            "safety check must stop rather than pass.")


@dataclass(frozen=True)
class ToleranceConfig:
    """Per-joint tolerances plus persistence. No defaults anywhere."""
    per_joint_deg: tuple[float, ...]
    severe_multiplier: float
    persistence_cycles: int
    source: str = "deployment_config"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ToleranceConfig":
        missing: list[str] = []
        pj = cfg.get("commanded_observed_tolerance_deg")
        if pj in (None, "", [], {}):
            missing.append("commanded_observed_tolerance_deg")
            pj = None
        elif isinstance(pj, (int, float)) and not isinstance(pj, bool):
            pj = tuple(float(pj) for _ in range(ARM_DIM))   # one value for all
        else:
            seq = list(pj)
            if len(seq) != ARM_DIM:
                missing.append(f"commanded_observed_tolerance_deg (need {ARM_DIM})")
                pj = None
            else:
                pj = tuple(float(x) for x in seq)
        sev = cfg.get("tolerance_severe_multiplier")
        if sev in (None, "", [], {}):
            missing.append("tolerance_severe_multiplier")
        per = cfg.get("tolerance_persistence_cycles")
        if per in (None, "", [], {}):
            missing.append("tolerance_persistence_cycles")
        if missing:
            raise ToleranceMissing(missing)
        return cls(pj, float(sev), int(per), str(cfg.get("source", "deployment_config")))

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d


@dataclass
class CycleReport:
    cycle: int
    ipoc: int | None
    commanded: list[float]
    interpolated: list[float]
    measured: list[float]
    residual: list[float]
    worst_joint: str | None
    worst_residual: float
    tolerance: float
    verdict: Verdict
    reason: str
    consecutive_breaches: int = 0

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["verdict"] = self.verdict.value
        for k in ("commanded", "interpolated", "measured", "residual"):
            d[k] = [round(float(v), 4) for v in d[k]]
        d["worst_residual"] = round(self.worst_residual, 5)
        return d


class DeviationMonitor:
    """Stateful across cycles, because persistence is the whole point."""

    def __init__(self, tolerances: ToleranceConfig) -> None:
        self.tol = tolerances
        self.cycle = 0
        self.consecutive = 0
        self.latched: str | None = None
        self.history: list[CycleReport] = []

    @property
    def faulted(self) -> bool:
        return self.latched is not None

    def observe(self, *, commanded: Sequence[float], interpolated: Sequence[float],
                measured: Sequence[float], ipoc: int | None = None,
                keep_history: int = 512) -> CycleReport:
        self.cycle += 1
        cmd = [float(v) for v in list(commanded)[:ARM_DIM]]
        itp = [float(v) for v in list(interpolated)[:ARM_DIM]]
        msd = [float(v) for v in list(measured)[:ARM_DIM]]
        # Residual is measured against what was ACTUALLY sent this cycle -- the
        # interpolated point -- not the chunk's endpoint, which the arm is not
        # yet supposed to have reached.
        resid = [msd[j] - itp[j] for j in range(ARM_DIM)]

        worst_j, worst = 0, 0.0
        over = False
        severe = False
        for j in range(ARM_DIM):
            a = abs(resid[j])
            if a > worst:
                worst, worst_j = a, j
            if a > self.tol.per_joint_deg[j]:
                over = True
                if a > self.tol.per_joint_deg[j] * self.tol.severe_multiplier:
                    severe = True

        if over:
            self.consecutive += 1
        else:
            self.consecutive = 0

        if self.latched:
            verdict, reason = Verdict.FAULT, f"latched: {self.latched}"
        elif severe:
            self.latched = (f"severe deviation on {ACTION_NAMES[worst_j]}: "
                            f"{worst:.4f} deg > "
                            f"{self.tol.per_joint_deg[worst_j] * self.tol.severe_multiplier:.4f}")
            verdict, reason = Verdict.FAULT, self.latched
        elif self.consecutive >= self.tol.persistence_cycles:
            self.latched = (f"deviation persisted {self.consecutive} cycles "
                            f"(limit {self.tol.persistence_cycles}), worst "
                            f"{ACTION_NAMES[worst_j]} {worst:.4f} deg")
            verdict, reason = Verdict.FAULT, self.latched
        elif over:
            verdict = Verdict.HOLD
            reason = (f"isolated deviation on {ACTION_NAMES[worst_j]}: {worst:.4f} "
                      f"deg > {self.tol.per_joint_deg[worst_j]:.4f} "
                      f"({self.consecutive}/{self.tol.persistence_cycles} cycles)")
        else:
            verdict, reason = Verdict.OK, "within tolerance"

        rep = CycleReport(self.cycle, ipoc, cmd, itp, msd, resid,
                          ACTION_NAMES[worst_j], worst,
                          self.tol.per_joint_deg[worst_j], verdict, reason,
                          self.consecutive)
        self.history.append(rep)
        if len(self.history) > keep_history:
            self.history = self.history[-keep_history:]
        return rep

    def summary(self) -> dict[str, Any]:
        counts = {v.value: 0 for v in Verdict}
        for r in self.history:
            counts[r.verdict.value] += 1
        return {"schema": SCHEMA, "cycles": self.cycle, "verdicts": counts,
                "latched": self.latched, "faulted": self.faulted,
                "tolerances": self.tol.to_log(),
                "worst_seen": round(max((r.worst_residual for r in self.history),
                                        default=0.0), 5)}
