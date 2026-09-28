"""Live frames from the three YAM cameras, on the KUKA branch's grabber.

`kuka.cameras` already has the part that matters -- one background thread per
camera, newest frame under a lock, monotonic grab time on every frame, JPEG at
a pinned quality, and a snapshot that REFUSES a stale or missing set. That is
reused as is. Two things are YAM-specific:

  names and order   top, left, right -- the checkpoint's training order
  frame size        640x360 by default. The checkpoint was trained on 360x640
                    frames and openpi pads to 224x224 preserving aspect, so a
                    4:3 capture would show the policy letterbox bars it never
                    saw. Capturing at another 16:9 size is fine.

The KUKA cell's Tera-camera rule (nodes 1 and 3 are metadata and refused) is
NOT carried over: it describes that cell's hardware, not these cameras. Here the
mapping must simply be given explicitly, and confirmed by looking.

NO CAPTURE HAPPENS ON IMPORT.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..kuka.cameras import (DEFAULT_MAX_AGE_S, ENCODE_MIME, CameraError, Frame,
                            _Grabber, frames_to_data_urls, frames_to_packet_refs)
from .contract import CAMERA_NAMES, IMAGE_SOURCE_HW

SCHEMA = "hybrid_rollout.robodojo.yam.cameras.v1"
HEIGHT, WIDTH = IMAGE_SOURCE_HW
FPS = 30

__all__ = ["CAMERA_NAMES", "CameraError", "DEFAULT_MAX_AGE_S", "ENCODE_MIME",
           "Frame", "LiveCameras", "frames_to_data_urls", "frames_to_packet_refs",
           "parse_mapping"]


def parse_mapping(text: str) -> dict[str, int]:
    """'top:4,left:0,right:2' -> {'top': 4, 'left': 0, 'right': 2}. Strict."""
    out: dict[str, int] = {}
    for pair in (text or "").split(","):
        if not pair.strip():
            continue
        name, sep, node = pair.partition(":")
        name = name.strip()
        if not sep or name not in CAMERA_NAMES:
            raise CameraError(f"bad camera entry {pair!r}; names are {CAMERA_NAMES}")
        if name in out:
            raise CameraError(f"camera {name!r} given twice")
        out[name] = int(node)
    if len(set(out.values())) != len(out):
        raise CameraError(f"two cameras share a device node: {out}")
    return out


class LiveCameras:
    """Three live cameras. Opens nothing until start()."""

    def __init__(self, mapping: dict[str, int], *,
                 names: Sequence[str] = CAMERA_NAMES,
                 width: int = WIDTH, height: int = HEIGHT, fps: int = FPS) -> None:
        self.names = tuple(names)
        self.mapping = dict(mapping)
        self.width, self.height, self.fps = width, height, fps
        self._grabbers: dict[str, _Grabber] = {}

    def start(self) -> "LiveCameras":
        missing = [n for n in self.names if n not in self.mapping]
        if missing:
            raise CameraError(
                f"no device index for {missing}. Give an explicit mapping such as "
                f"top:4,left:0,right:2; device order is not stable across "
                f"reboots, and a swapped wrist pair is silent.")
        if abs(self.width / self.height - WIDTH / HEIGHT) > 0.01:
            raise CameraError(
                f"{self.width}x{self.height} is not 16:9; the checkpoint was "
                f"trained on {WIDTH}x{HEIGHT} frames and would see different "
                f"padding")
        for name in self.names:
            g = _Grabber(self.mapping[name], name, self.width, self.height, self.fps)
            g.start()
            self._grabbers[name] = g
        return self

    def snapshot(self, max_age_s: float = DEFAULT_MAX_AGE_S) -> dict[str, Frame]:
        """Newest frame per camera. Raises if any is missing or stale."""
        import time
        now = time.monotonic()
        out: dict[str, Frame] = {}
        problems: list[str] = []
        for name in self.names:
            g = self._grabbers.get(name)
            f = g.latest() if g else None
            if f is None:
                problems.append(f"{name}: no frame yet")
                continue
            age = f.age_s(now)
            if age > max_age_s:
                problems.append(f"{name}: {age:.3f}s old (limit {max_age_s}s)")
                continue
            if (f.height, f.width) != (self.height, self.width):
                problems.append(f"{name}: device delivered {f.width}x{f.height}, "
                                f"asked for {self.width}x{self.height}")
                continue
            out[name] = f
        if problems:
            raise CameraError("stale, missing or mis-sized frames -- refusing to "
                              "build an observation: " + "; ".join(problems))
        return out

    def stop(self) -> None:
        for g in self._grabbers.values():
            g.stop()

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "names": list(self.names), "mapping": self.mapping,
                "size": [self.width, self.height],
                "cameras": [g.to_log() for g in self._grabbers.values()]}
