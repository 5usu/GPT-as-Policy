"""Sampling a recorded episode: synchronized frames + state + demo trajectory.

Phase 1 of the dishwasher experiment shows Astra a recorded trajectory and the
observation videos that go with it. That requires pairing a control tick with
the camera frame taken at the same instant -- which is only meaningful if the
manifest's alignment checks passed. `RecordedEpisode.open` refuses otherwise,
because a review of mismatched frames and states reviews nothing.

MEDIA LIVE OUTSIDE THE REPOSITORY. The manifest references videos and state by
relative path plus sha256; this module resolves them against a media root the
caller supplies. Nothing here commits, copies or embeds media.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .contract import ACTION_DIM, ARM_DIM, CONTROL_HZ
from .experiment import (EpisodeManifest, ManifestError, frame_for_tick,
                         load_manifest, validate_manifest)

SCHEMA = "hybrid_rollout.robodojo.kuka.episode.v1"


@dataclass
class Sample:
    """One reviewable moment: a tick, its frames, its state, its next chunk."""
    tick: int
    observation_id: str
    epoch: float
    state: list[float]
    frames: dict[str, Any]              # camera -> {frame_index, path|missing}
    recorded_chunk: list[list[float]]   # the demo's next N commanded targets
    chunk_provenance: str = "recorded_demo"

    def to_log(self) -> dict[str, Any]:
        return {"tick": self.tick, "observation_id": self.observation_id,
                "epoch": self.epoch, "frames": self.frames,
                "n_chunk": len(self.recorded_chunk),
                "chunk_provenance": self.chunk_provenance}


class RecordedEpisode:
    """A validated recorded episode, sampleable at control ticks."""

    def __init__(self, manifest: EpisodeManifest, *, media_root: Path | None,
                 state_rows: list[list[float]] | None = None,
                 traj_rows: list[list[float]] | None = None) -> None:
        self.m = manifest
        self.media_root = Path(media_root) if media_root else None
        self._state = state_rows
        self._traj = traj_rows

    @classmethod
    def open(cls, manifest_path: str | Path, *, media_root: str | Path | None = None,
             state_rows: list[list[float]] | None = None,
             traj_rows: list[list[float]] | None = None,
             require_valid: bool = True) -> "RecordedEpisode":
        m = load_manifest(manifest_path)
        problems = validate_manifest(m)
        if problems and require_valid:
            raise ManifestError(
                "episode manifest failed validation; refusing to sample it "
                "because frames and state rows may not describe the same "
                "instants:\n  - " + "\n  - ".join(problems))
        return cls(m, media_root=Path(media_root) if media_root else None,
                   state_rows=state_rows, traj_rows=traj_rows)

    # ------------------------------------------------------------------ rows
    @property
    def n_ticks(self) -> int:
        return int((self.m.raw.get("state") or {}).get("n_rows") or 0)

    @property
    def control_hz(self) -> float:
        return float((self.m.raw.get("timestamps") or {}).get("control_hz")
                     or CONTROL_HZ)

    def state_at(self, tick: int) -> list[float]:
        if self._state is None:
            raise ManifestError(
                "state rows not loaded. Pass state_rows= (parquet/npz decoding is "
                "deliberately out of scope here: this package does not depend on "
                "a dataframe stack so it can run on the Jetson).")
        if not 0 <= tick < len(self._state):
            raise ManifestError(f"tick {tick} outside 0..{len(self._state) - 1}")
        row = list(self._state[tick])
        if len(row) != ACTION_DIM:
            raise ManifestError(f"state row {tick} has {len(row)} dims")
        return row

    def chunk_at(self, tick: int, n: int) -> list[list[float]]:
        """The demonstration's next `n` commanded targets. Recorded, not model."""
        if self._traj is None:
            raise ManifestError("trajectory rows not loaded. Pass traj_rows=.")
        end = min(tick + n, len(self._traj))
        if tick >= end:
            raise ManifestError(f"no trajectory rows at tick {tick}")
        return [list(r) for r in self._traj[tick:end]]

    # ---------------------------------------------------------------- frames
    def frames_at(self, tick: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for v in self.m.videos:
            cam = str(v.get("camera"))
            try:
                idx = frame_for_tick(self.m, cam, tick)
            except ManifestError as exc:
                out[cam] = {"error": str(exc)}
                continue
            entry: dict[str, Any] = {"frame_index": idx, "video": v.get("path"),
                                     "video_sha256": v.get("sha256")}
            if self.media_root is not None:
                png = self.media_root / "frames" / f"{cam}_{idx:06d}.png"
                entry["frame_path"] = str(png)
                entry["frame_present"] = png.exists()
            else:
                entry["frame_present"] = False
                entry["note"] = "no media_root supplied; frame not resolved"
            out[cam] = entry
        return out

    # --------------------------------------------------------------- sampling
    def sample(self, tick: int, *, chunk_steps: int = 50) -> Sample:
        ts = self.m.raw.get("timestamps") or {}
        epoch = float(ts.get("recorded_start_epoch") or 0.0) + tick / self.control_hz
        return Sample(
            tick=tick,
            observation_id=f"{self.m.episode_id}:t{tick:06d}",
            epoch=epoch, state=self.state_at(tick), frames=self.frames_at(tick),
            recorded_chunk=self.chunk_at(tick, chunk_steps))

    def sample_ticks(self, *, every: int, chunk_steps: int = 50,
                     limit: int | None = None) -> list[Sample]:
        """Evenly spaced ticks. `every` is in control ticks, not frames."""
        if every <= 0:
            raise ValueError("every must be positive")
        ticks = list(range(0, max(0, self.n_ticks - chunk_steps), every))
        if limit is not None:
            ticks = ticks[:limit]
        return [self.sample(t, chunk_steps=chunk_steps) for t in ticks]

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **self.m.to_log(), "n_ticks": self.n_ticks,
                "control_hz": self.control_hz}
