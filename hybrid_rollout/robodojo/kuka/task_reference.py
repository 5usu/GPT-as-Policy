"""What success looked like once, as context for the reviewer.

WHY THIS EXISTS
  Until now Astra judged every proposal against an expectation it invented on
  the spot. The schema asks for `expected_next_intent` and
  `predicted_next_intent` and then compares them -- but BOTH come from Astra, so
  `intent_status` was self-referential, and `intent_status=misaligned` is one of
  only two takeover triggers. The task string plus two camera views was the
  whole basis for deciding what should happen next.

  A reference episode gives that comparison an anchor: this is one run that
  actually succeeded at this task on this robot.

WHY IT IS A SUMMARY AND NOT A TRAJECTORY
  Deliberately. Handing over 50x7 joint arrays from a successful run invites two
  failures that are worse than the problem being fixed:

    1. The reviewer copies them. An `edit` is supposed to be a bounded
       correction to THIS proposal from THIS pose; replaying another episode's
       absolute joint targets is not a correction, it is a different episode's
       motion commanded from the wrong starting pose. Raw rows are therefore
       never rendered, so there is nothing to copy.

    2. The reviewer treats divergence as failure. The reference succeeded with
       one particular object placement, approach angle and timing. A correct
       proposal from a different starting pose will not match it. If the
       reference reads as a template, every legitimate variation becomes
       "misaligned" -- which would manufacture takeovers rather than catch them.

  So what crosses over is the SHAPE of a successful attempt: the ordered
  subgoals, where the gripper opened and closed, roughly how long each phase
  took, and the net joint travel. That is what `expected_next_intent` needs.
  It is not enough to replay, and that is the point.

VERIFICATION IS REQUIRED
  A reference is only meaningful if somebody confirmed the episode actually
  succeeded. `verified_successful` plus a named verifier are mandatory; an
  unverified episode is refused rather than described as a success.

COMPARABILITY
  Upstream GPT-as-Policy gives its reviewer no reference. A run WITH one is
  therefore not comparable to the published 48% / 62.60 hybrid numbers. This is
  off by default and recorded in the audit so the two cannot be mixed up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .contract import ARM_DIM, CONTROL_HZ

SCHEMA = "kuka.task_reference.v1"

#: Gripper crossing point for detecting a grasp or release.
GRIP_THRESHOLD = 0.5

#: Never rendered into a packet. Named so the intent is explicit in code review.
WITHHELD_FROM_PROMPT = ("raw joint rows", "per-step targets")


class ReferenceRefused(ValueError):
    """The episode cannot be presented to a reviewer as a success."""


@dataclass
class Phase:
    """One segment of a successful attempt, split at gripper transitions."""
    name: str
    start_step: int
    n_steps: int
    net_joint_deg: list[float]
    gripper_from: float
    gripper_to: float

    @property
    def duration_s(self) -> float:
        return self.n_steps / CONTROL_HZ

    def render(self) -> str:
        travel = ", ".join(f"A{j+1}{v:+.1f}" for j, v in
                           enumerate(self.net_joint_deg) if abs(v) >= 0.5)
        return (f"  {self.name:<18} {self.duration_s:4.2f} s  "
                f"grip {self.gripper_from:.2f}->{self.gripper_to:.2f}  "
                f"[{travel or 'no joint travel above 0.5 deg'}]")


@dataclass
class TaskReference:
    """A verified successful episode, summarised for the reviewer."""
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
                f"episode {self.episode_id!r} is not marked successful. A "
                f"reference tells the reviewer 'this is what success looks "
                f"like'; an unverified episode would make that a claim nobody "
                f"checked.")
        if not self.verified_by.strip():
            raise ReferenceRefused(
                f"episode {self.episode_id!r} has no named verifier. Who "
                f"confirmed it succeeded is part of the evidence.")
        if not self.task.strip():
            raise ReferenceRefused("a reference needs the task it succeeded at")

    @property
    def duration_s(self) -> float:
        return self.n_steps / CONTROL_HZ

    def render(self) -> str:
        """The packet block. Leads and closes with the scene caveat."""
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
            lines += [f"    {i+1}. {s}" for i, s in enumerate(self.subgoals)]
        if self.phases:
            lines.append("  phases (split at gripper transitions):")
            lines += [p.render() for p in self.phases]
        if self.note:
            lines.append(f"  note      : {self.note}")
        lines += [
            "",
            "HOW TO USE IT. Use it to decide what the NEXT SUBGOAL should be, "
            "which is what expected_next_intent is asking for.",
            "Object placement, approach angle and timing DIFFER from the run "
            "above. A correct proposal from the current pose will not match it.",
            "Divergence from this reference is NOT evidence of failure and is "
            "NOT a takeover reason on its own.",
            "Its joint values are deliberately not shown: they were valid from "
            "a different starting pose, so they are not a correction to this "
            "proposal and must not be reproduced as one.",
        ]
        return "\n".join(lines)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["duration_s"] = round(self.duration_s, 3)
        d["withheld_from_prompt"] = list(WITHHELD_FROM_PROMPT)
        return d


def segment_phases(rows: Sequence[Sequence[float]]) -> list[Phase]:
    """Split a recorded episode at gripper transitions.

    Gripper state is the one unambiguous task landmark available without a
    scene model: closing is a grasp attempt, opening is a release. Everything
    between is approach, transport or retreat.
    """
    rows = [list(r) for r in rows]
    if not rows:
        return []
    has_grip = len(rows[0]) > ARM_DIM
    marks = [0]
    if has_grip:
        for i in range(1, len(rows)):
            prev, cur = float(rows[i - 1][ARM_DIM]), float(rows[i][ARM_DIM])
            crossed = ((cur >= GRIP_THRESHOLD > prev)
                       or (cur < GRIP_THRESHOLD <= prev))
            if crossed:
                marks.append(i)
    marks.append(len(rows))

    out: list[Phase] = []
    seen_closed = False
    for k in range(len(marks) - 1):
        a, b = marks[k], marks[k + 1]
        seg = rows[a:b]
        if not seg:
            continue
        closed = has_grip and float(seg[0][ARM_DIM]) >= GRIP_THRESHOLD
        if not has_grip:
            name = f"segment {k + 1}"
        elif closed:
            name = "holding (closed)"
            seen_closed = True
        elif seen_closed:
            # open again after having held something: this is the release side
            name = "retreat (released)"
        else:
            name = "approach (open)"
        net = [round(float(seg[-1][j]) - float(seg[0][j]), 2)
               for j in range(ARM_DIM)]
        g0 = float(seg[0][ARM_DIM]) if has_grip else 0.0
        g1 = float(seg[-1][ARM_DIM]) if has_grip else 0.0
        out.append(Phase(name, a, len(seg), net, round(g0, 3), round(g1, 3)))
    return out


def from_episode(*, task: str, episode_id: str, rows: Sequence[Sequence[float]],
                 verified_by: str, verified_successful: bool = False,
                 subgoals: Sequence[str] | None = None,
                 note: str = "") -> TaskReference:
    """Summarise a recorded episode as a reference. Refuses if unverified."""
    return TaskReference(
        task=task, episode_id=episode_id,
        verified_successful=verified_successful, verified_by=verified_by,
        n_steps=len(rows), phases=segment_phases(rows),
        subgoals=list(subgoals or []), note=note)


def load(path: str) -> TaskReference:
    """Load a reference from JSON written by the operator.

    Shape:
      {"task": ..., "episode_id": ..., "verified_successful": true,
       "verified_by": ..., "subgoals": [...], "note": ...,
       "rows": [[a1..a6, grip], ...]}   # rows are summarised, never forwarded
    """
    import json
    from pathlib import Path
    raw = json.loads(Path(path).read_text())
    missing = [k for k in ("task", "episode_id", "verified_by")
               if not str(raw.get(k, "")).strip()]
    if missing:
        raise ReferenceRefused(
            f"{path}: reference is missing {', '.join(missing)}")
    return from_episode(
        task=str(raw["task"]), episode_id=str(raw["episode_id"]),
        rows=raw.get("rows") or [],
        verified_by=str(raw["verified_by"]),
        verified_successful=bool(raw.get("verified_successful", False)),
        subgoals=raw.get("subgoals") or [], note=str(raw.get("note", "")))
