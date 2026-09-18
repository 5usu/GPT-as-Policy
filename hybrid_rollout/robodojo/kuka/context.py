"""The evidence Astra actually needs, and the record of what it was given.

WHY THIS EXISTS
Every live call so far returned execution_status=uncertain with the same stated
reason: no prior observation, so gate question 1 ("what happened during the LAST
executed chunk") had nothing to work on. The upstream gate permits a takeover
only on execution_status=failed or intent_status=misaligned, so with question 1
permanently unanswerable a correction could never be reached. That is very
likely why we sit at 0% corrections against upstream's 14.4% -- a missing
channel, not a deficient reviewer.

A cycle therefore carries:
    before   frames from the PREVIOUS cycle, before its chunk was executed
    current  frames now
    after    frames once the executed prefix has run  (same as the next before)
    executed the prefix that actually ran, not the whole proposed chunk
    feedback measured joint positions after execution
    previous what the reviewer decided last time
    progress the task_progress it maintained

ATTRIBUTION IS NOT OPTIONAL
`before` and `after` bracket the EXECUTED PREFIX. They are evidence about that
prefix and nothing else. The proposed chunk under review has not run, so no image
shows its result. `CycleContext` refuses to present frames as bracketing a chunk
that was never executed, because a reviewer that credits an unexecuted proposal
with a visible outcome is not reviewing it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.context.v1"


@dataclass
class FrameRef:
    camera: str
    kind: str                       # before | current | after
    frame_index: int | None = None
    path: str | None = None
    epoch: float | None = None
    live: bool = False

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutedPrefix:
    """What actually ran. Never the whole proposed chunk."""
    rows: list[list[float]] = field(default_factory=list)
    n_steps: int = 0
    started_epoch: float | None = None
    finished_epoch: float | None = None
    source: str = "unknown"

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["values_shown"] = self.rows[:5]
        d.pop("rows", None)
        return d


@dataclass
class CycleContext:
    """One reviewable cycle with its full evidence and history."""
    cycle: int
    observation_id: str
    task: str
    state: list[float]
    proposed_chunk: list[list[float]]
    frames: list[FrameRef] = field(default_factory=list)
    executed: ExecutedPrefix | None = None
    measured_feedback: list[float] | None = None
    previous_decision: dict[str, Any] | None = None
    task_progress: dict[str, Any] | None = None
    epoch: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        kinds = {f.kind for f in self.frames}
        bad = kinds - {"before", "current", "after"}
        if bad:
            raise ValueError(f"unknown frame kind(s): {sorted(bad)}")
        if ("before" in kinds or "after" in kinds) and self.executed is None:
            raise ValueError(
                "before/after frames bracket an EXECUTED prefix. None was "
                "supplied, so these frames cannot be presented as evidence of "
                "anything having run -- the proposed chunk has not executed and "
                "no image shows its result.")

    @property
    def has_execution_evidence(self) -> bool:
        kinds = {f.kind for f in self.frames}
        return bool(self.executed and self.executed.n_steps
                    and "before" in kinds and "after" in kinds)

    def evidence_summary(self) -> str:
        """Prose block for the review packet. States what each frame proves."""
        if not self.has_execution_evidence:
            return ("NO EXECUTION EVIDENCE THIS CYCLE. Nothing has been executed "
                    "since the last observation, so the last-chunk question "
                    "cannot be answered from images. Say uncertain rather than "
                    "inferring an outcome.")
        ex = self.executed
        lines = [
            "EXECUTION EVIDENCE (ground truth, already executed):",
            f"  the BEFORE frames and the AFTER frames bracket {ex.n_steps} step(s)",
            f"  that were actually executed ({ex.source}). The visible change",
            "  between them IS the outcome of those steps, and of nothing else.",
        ]
        if self.measured_feedback:
            lines.append("  measured joints after execution: "
                         f"{[round(float(v), 2) for v in self.measured_feedback[:6]]}")
            if ex.rows:
                cmd = ex.rows[-1]
                drift = [round(float(self.measured_feedback[j]) - float(cmd[j]), 3)
                         for j in range(min(6, len(cmd)))]
                lines.append(f"  commanded-minus-measured on the last step: {drift}")
        lines.append("  The chunk under review below has NOT been executed; no "
                     "image shows its result.")
        return "\n".join(lines)

    def history_summary(self) -> str:
        if not self.previous_decision and not self.task_progress:
            return ""
        out = ["PRIOR CYCLE:"]
        if self.previous_decision:
            d = self.previous_decision
            out.append(f"  your last decision: mode={d.get('mode')} "
                       f"steps={d.get('steps')}")
            if d.get("reason"):
                out.append(f"  your stated reason: {str(d['reason'])[:200]}")
        if self.task_progress:
            tp = self.task_progress
            out.append(f"  verified_completed: {tp.get('verified_completed')}")
            out.append(f"  currently_attempting: {tp.get('currently_attempting')}")
            out.append(f"  remaining: {tp.get('remaining')}")
        return "\n".join(out)

    def frame_refs(self) -> dict[str, Any]:
        """Shape `packet.build_packet` wants, labelled by kind."""
        return {f"{f.kind}/{f.camera}": {"frame_index": f.frame_index,
                                         "frame_path": f.path,
                                         "frame_present": f.path is not None or f.live,
                                         "kind": f.kind, "live": f.live}
                for f in self.frames}

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "cycle": self.cycle,
                "observation_id": self.observation_id, "task": self.task,
                "epoch": self.epoch,
                "state": [round(float(v), 4) for v in self.state],
                "proposed_n_steps": len(self.proposed_chunk),
                "frames": [f.to_log() for f in self.frames],
                "executed": self.executed.to_log() if self.executed else None,
                "measured_feedback": self.measured_feedback,
                "previous_decision_mode": (self.previous_decision or {}).get("mode"),
                "task_progress": self.task_progress,
                "has_execution_evidence": self.has_execution_evidence}


class CycleHistory:
    """Carries state between cycles so question 1 becomes answerable."""

    def __init__(self) -> None:
        self.last_frames: list[FrameRef] = []
        self.last_decision: dict[str, Any] | None = None
        self.last_progress: dict[str, Any] | None = None
        self.last_executed: ExecutedPrefix | None = None
        self.cycle = 0

    def begin(self, *, observation_id: str, task: str, state: Sequence[float],
              proposed_chunk: Sequence[Sequence[float]],
              current_frames: Sequence[FrameRef],
              measured_feedback: Sequence[float] | None = None) -> CycleContext:
        self.cycle += 1
        frames = [FrameRef(f.camera, "before", f.frame_index, f.path, f.epoch, f.live)
                  for f in self.last_frames] if self.last_executed else []
        frames += [FrameRef(f.camera, "after", f.frame_index, f.path, f.epoch, f.live)
                   for f in current_frames] if self.last_executed else []
        frames += [FrameRef(f.camera, "current", f.frame_index, f.path, f.epoch,
                            f.live) for f in current_frames]
        return CycleContext(
            cycle=self.cycle, observation_id=observation_id, task=task,
            state=[float(v) for v in state],
            proposed_chunk=[list(r) for r in proposed_chunk],
            frames=frames, executed=self.last_executed,
            measured_feedback=list(measured_feedback) if measured_feedback else None,
            previous_decision=self.last_decision, task_progress=self.last_progress)

    def commit(self, *, frames: Sequence[FrameRef], decision: dict[str, Any] | None,
               executed: ExecutedPrefix | None) -> None:
        """Record what this cycle ended with, for the next one's `before`."""
        self.last_frames = list(frames)
        self.last_decision = dict(decision) if decision else None
        if decision and isinstance(decision.get("assessment"), dict):
            self.last_progress = decision["assessment"].get("task_progress")
        self.last_executed = executed


class RunStore:
    """Append-only persistence of everything a cycle produced.

    Separate files per kind so a proposal can never be silently confused with a
    decision or with a validation result when the run is read back.
    """

    KINDS = ("observation", "proposal", "context", "decision", "validation",
             "execution")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = {k: 0 for k in self.KINDS}

    def _write(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind not in self.KINDS:
            raise ValueError(f"unknown record kind {kind!r}")
        row = {"kind": kind, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               **payload}
        with (self.root / f"{kind}.jsonl").open("a") as f:   # append-only
            f.write(json.dumps(row, default=str) + "\n")
        self.counts[kind] += 1
        return row

    def observation(self, **kw): return self._write("observation", kw)
    def proposal(self, **kw): return self._write("proposal", kw)
    def context(self, ctx: CycleContext): return self._write("context", ctx.to_log())
    def decision(self, **kw): return self._write("decision", kw)
    def validation(self, **kw): return self._write("validation", kw)
    def execution(self, **kw): return self._write("execution", kw)

    def summary(self) -> dict[str, Any]:
        return {"root": str(self.root), "counts": dict(self.counts)}
