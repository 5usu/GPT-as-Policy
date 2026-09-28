"""A verified successful YAM episode, summarised as context for the reviewer.

Same contract as `kuka.task_reference`, for the same reasons: the reviewer gets
the SHAPE of a success -- ordered subgoals, where each gripper opened and
closed, how long each phase took, net joint travel per arm -- and never the raw
joint rows, so there is nothing to copy and no template to punish legitimate
variation against. Verification by a named person is mandatory, and a run with
a reference is not comparable to upstream's published numbers.

Two things differ from the KUKA version:
  - phases split at a transition of EITHER gripper, and are labelled by the
    state of both hands;
  - closed is BELOW the threshold (YAM: 0 closed, 1 open).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..kuka.task_reference import WITHHELD_FROM_PROMPT, ReferenceRefused
from .contract import (ARM_JOINT_INDICES, ARMS, CONTROL_HZ, GRIPPER_INDICES,
                       GRIPPER_THRESHOLD)

SCHEMA = "yam.task_reference.v1"

__all__ = ["Phase", "ReferenceRefused", "TaskReference", "from_episode", "load",
           "segment_phases"]


def _closed(value: float) -> bool:
    return float(value) < GRIPPER_THRESHOLD


@dataclass
class Phase:
    name: str
    start_step: int
    n_steps: int
    net_joint_rad: dict[str, list[float]]
    grippers_from: dict[str, float]
    grippers_to: dict[str, float]

    @property
    def duration_s(self) -> float:
        return self.n_steps / CONTROL_HZ

    def render(self) -> str:
        parts = []
        for arm in ARMS:
            travel = ", ".join(f"j{j + 1}{v:+.2f}" for j, v in
                               enumerate(self.net_joint_rad[arm]) if abs(v) >= 0.01)
            parts.append(f"{arm} grip {self.grippers_from[arm]:.2f}->"
                         f"{self.grippers_to[arm]:.2f} "
                         f"[{travel or 'no joint travel above 0.01 rad'}]")
        return f"  {self.name:<26} {self.duration_s:4.2f} s  " + "; ".join(parts)


@dataclass
class TaskReference:
    task: str
    episode_id: str
    verified_successful: bool
    verified_by: str
    n_steps: int
    phases: list[Phase] = field(default_factory=list)
    subgoals: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.verified_successful:
            raise ReferenceRefused(
                f"episode {self.episode_id!r} is not marked successful; an "
                f"unverified episode cannot be presented as what success looks like")
        if not self.verified_by.strip():
            raise ReferenceRefused(
                f"episode {self.episode_id!r} has no named verifier")
        if not self.task.strip():
            raise ReferenceRefused("a reference needs the task it succeeded at")

    @property
    def duration_s(self) -> float:
        return self.n_steps / CONTROL_HZ

    def render(self) -> str:
        lines = [
            "TASK REFERENCE -- A DIFFERENT EPISODE THAT SUCCEEDED AT THIS TASK.",
            "This is NOT the current scene and NOT a trajectory to reproduce.",
            f"  task      : {self.task}",
            f"  episode   : {self.episode_id} "
            f"({self.n_steps} steps, {self.duration_s:.2f} s)",
            f"  confirmed : successful, verified by {self.verified_by}",
        ]
        if self.subgoals:
            lines.append("  subgoals in the order they were achieved:")
            lines += [f"    {i + 1}. {s}" for i, s in enumerate(self.subgoals)]
        if self.phases:
            lines.append("  phases (split at gripper transitions, either arm):")
            lines += [p.render() for p in self.phases]
        if self.note:
            lines.append(f"  note      : {self.note}")
        lines += [
            "",
            "HOW TO USE IT. Use it to decide what the NEXT SUBGOAL should be, "
            "which is what expected_next_intent is asking for.",
            "Object placement, approach and timing DIFFER from the run above. "
            "Divergence from this reference is NOT evidence of failure and is "
            "NOT a takeover reason on its own.",
            "Its joint values are deliberately not shown and must not be "
            "reproduced as a correction.",
        ]
        return "\n".join(lines)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["duration_s"] = round(self.duration_s, 3)
        d["withheld_from_prompt"] = list(WITHHELD_FROM_PROMPT)
        return d


def _hands(row: Sequence[float]) -> str:
    return ", ".join(f"{arm} {'closed' if _closed(row[GRIPPER_INDICES[arm]]) else 'open'}"
                     for arm in ARMS)


def segment_phases(rows: Sequence[Sequence[float]]) -> list[Phase]:
    rows = [list(r) for r in rows]
    if not rows:
        return []
    marks = [0]
    for i in range(1, len(rows)):
        if any(_closed(rows[i][g]) != _closed(rows[i - 1][g])
               for g in GRIPPER_INDICES.values()):
            marks.append(i)
    marks.append(len(rows))
    out: list[Phase] = []
    for k in range(len(marks) - 1):
        seg = rows[marks[k]:marks[k + 1]]
        if not seg:
            continue
        net = {arm: [round(float(seg[-1][j]) - float(seg[0][j]), 3)
                     for j in ARM_JOINT_INDICES[arm]] for arm in ARMS}
        out.append(Phase(
            f"{k + 1}: {_hands(seg[0])}", marks[k], len(seg), net,
            {arm: round(float(seg[0][g]), 3) for arm, g in GRIPPER_INDICES.items()},
            {arm: round(float(seg[-1][g]), 3) for arm, g in GRIPPER_INDICES.items()}))
    return out


def from_episode(*, task: str, episode_id: str, rows: Sequence[Sequence[float]],
                 verified_by: str, verified_successful: bool = False,
                 subgoals: Sequence[str] | None = None,
                 note: str = "") -> TaskReference:
    return TaskReference(
        task=task, episode_id=episode_id,
        verified_successful=verified_successful, verified_by=verified_by,
        n_steps=len(rows), phases=segment_phases(rows),
        subgoals=list(subgoals or []), note=note)


def load(path: str) -> TaskReference:
    """Same JSON shape as the KUKA reference, with 14-value rows."""
    import json
    from pathlib import Path
    raw = json.loads(Path(path).read_text())
    missing = [k for k in ("task", "episode_id", "verified_by")
               if not str(raw.get(k, "")).strip()]
    if missing:
        raise ReferenceRefused(f"{path}: reference is missing {', '.join(missing)}")
    return from_episode(
        task=str(raw["task"]), episode_id=str(raw["episode_id"]),
        rows=raw.get("rows") or [], verified_by=str(raw["verified_by"]),
        verified_successful=bool(raw.get("verified_successful", False)),
        subgoals=raw.get("subgoals") or [], note=str(raw.get("note", "")))
