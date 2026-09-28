"""Sampling a recorded YAM episode: synchronized frames + state + demo actions.

`kuka.episode.RecordedEpisode` with the row width and the chunk length of this
checkpoint. The manifest format, its video/state alignment checks and the
tick-to-frame mapping are the KUKA ones, unchanged; see kuka/DEPLOYMENT.md.

    <episode_dir>/
      manifest.json                  # validates against the KUKA manifest checks
      state.json                     # [[14 values], ...] one row per control tick
      trajectory.json                # [[14 values], ...] recorded actions
      frames/<camera>_<frame:06d>.jpg|.png   # camera in top, left, right

A held-out MolmoAct2 YAM recording (e.g. allenai/19012026-block-13) converted
into this layout is how the reconstructed pi0.5 server is checked before any
arm is involved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..kuka.episode import RecordedEpisode, Sample
from .contract import ACTION_DIM, CHUNK_STEPS
from .experiment import ManifestError, frame_for_tick

__all__ = ["Sample", "YamRecordedEpisode"]


class YamRecordedEpisode(RecordedEpisode):
    def state_at(self, tick: int) -> list[float]:
        if self._state is None:
            raise ManifestError("state rows not loaded. Pass state_rows=.")
        if not 0 <= tick < len(self._state):
            raise ManifestError(f"tick {tick} outside 0..{len(self._state) - 1}")
        row = list(self._state[tick])
        if len(row) != ACTION_DIM:
            raise ManifestError(f"state row {tick} has {len(row)} dims, "
                                f"expected {ACTION_DIM}")
        return row

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
            if self.media_root is None:
                entry.update(frame_present=False,
                             note="no media_root supplied; frame not resolved")
            else:
                found = next((p for ext in (".jpg", ".png")
                              if (p := Path(self.media_root) / "frames" /
                                  f"{cam}_{idx:06d}{ext}").exists()), None)
                entry["frame_path"] = str(found) if found else None
                entry["frame_present"] = found is not None
            out[cam] = entry
        return out

    def sample(self, tick: int, *, chunk_steps: int = CHUNK_STEPS) -> Sample:
        return super().sample(tick, chunk_steps=chunk_steps)

    def sample_ticks(self, *, every: int, chunk_steps: int = CHUNK_STEPS,
                     limit: int | None = None) -> list[Sample]:
        return super().sample_ticks(every=every, chunk_steps=chunk_steps, limit=limit)
