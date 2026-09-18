"""Live observation frames. The piece that was missing for anything non-recorded.

Everything before this handled RECORDED frames: build_episode extracts them from
downloaded video, episode.py pairs them with control ticks, packet.py references
them by path. None of that helps in live_shadow or on a real cell, where the
frames have to come off the cameras as the arm moves. This module closes that
gap and nothing else.

MATCHED TO THE DEPLOYED STACK (KUKA/teleoperation/camera_capture.py)
  - OpenCV VideoCapture on /dev/videoN
  - 640x480 @ 30 fps, the resolution the checkpoint was trained on
  - cameras named `base` and `wrist`, becoming observation.images.{name}
  - background grab thread per camera, newest frame under a lock
Deviating from any of these would mean reviewing images the policy never saw in
that form.

FRESHNESS IS THE SAFETY PROPERTY HERE.
A stale frame is worse than no frame: it looks like a valid observation while
describing a pose the arm has already left. Every frame carries the monotonic
time it was grabbed, `LiveCameras.snapshot` refuses to return a set older than a
supplied bound, and a camera that has produced nothing is an error rather than a
silently absent key. `safety.check_freshness` then gates the observation itself.

NO CAPTURE HAPPENS ON IMPORT. Nothing opens a device until `start()`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.cameras.v1"

CAMERA_NAMES = ("base", "wrist")
WIDTH, HEIGHT, FPS = 640, 480, 30
DEFAULT_MAX_AGE_S = 0.25          # ~7 frames at 30 fps


class CameraError(RuntimeError):
    """A camera could not be opened, or produced nothing."""


@dataclass
class Frame:
    camera: str
    png: bytes | None                 # encoded, ready for a review packet
    grabbed_monotonic: float
    grabbed_epoch: float
    frame_id: int
    width: int = WIDTH
    height: int = HEIGHT

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.grabbed_monotonic

    def to_log(self) -> dict[str, Any]:
        return {"camera": self.camera, "frame_id": self.frame_id,
                "grabbed_epoch": self.grabbed_epoch,
                "width": self.width, "height": self.height,
                "bytes": len(self.png) if self.png else 0}


class CameraSource(Protocol):
    names: tuple[str, ...]

    def snapshot(self, max_age_s: float) -> dict[str, Frame]: ...


class _Grabber:
    """One camera, one background thread, newest frame under a lock."""

    def __init__(self, index: int, name: str, width: int = WIDTH,
                 height: int = HEIGHT, fps: int = FPS) -> None:
        self.index, self.name = index, name
        self.width, self.height, self.fps = width, height, fps
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._opened = False
        self._error: str | None = None

    def start(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"camera {self.name}: cannot open /dev/video{self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._opened = True
        self._thread = threading.Thread(target=self._loop, args=(cap,),
                                        name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def _loop(self, cap) -> None:
        import cv2
        n = 0
        try:
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    self._error = "read failed"
                    time.sleep(0.01)
                    continue
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    self._error = "png encode failed"
                    continue
                n += 1
                f = Frame(self.name, buf.tobytes(), time.monotonic(), time.time(),
                          n, img.shape[1], img.shape[0])
                with self._lock:
                    self._latest = f
                    self._error = None
        finally:
            cap.release()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def to_log(self) -> dict[str, Any]:
        f = self.latest()
        return {"camera": self.name, "device": f"/dev/video{self.index}",
                "opened": self._opened, "error": self._error,
                "latest": f.to_log() if f else None}


class LiveCameras:
    """The live source. Opens nothing until start()."""

    def __init__(self, mapping: dict[str, int] | None = None,
                 names: Sequence[str] = CAMERA_NAMES) -> None:
        self.names = tuple(names)
        self.mapping = dict(mapping or {})
        self._grabbers: dict[str, _Grabber] = {}

    def start(self) -> "LiveCameras":
        missing = [n for n in self.names if n not in self.mapping]
        if missing:
            raise CameraError(
                f"no device index for {missing}. Supply an explicit mapping such "
                f"as {{'base': 0, 'wrist': 2}}; device order is not stable across "
                f"reboots and guessing it can pair the wrong view with the wrong "
                f"camera name.")
        for name in self.names:
            g = _Grabber(self.mapping[name], name)
            g.start()
            self._grabbers[name] = g
        return self

    def snapshot(self, max_age_s: float = DEFAULT_MAX_AGE_S) -> dict[str, Frame]:
        """Newest frame per camera. Raises if any is missing or stale."""
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
            out[name] = f
        if problems:
            raise CameraError("stale or missing frames -- refusing to build an "
                              "observation: " + "; ".join(problems))
        return out

    def stop(self) -> None:
        for g in self._grabbers.values():
            g.stop()

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "names": list(self.names),
                "mapping": self.mapping,
                "cameras": [g.to_log() for g in self._grabbers.values()]}


class RecordedCameras:
    """Frames from an extracted episode. Lets the same code path run offline."""

    def __init__(self, frames_dir: str | Path, names: Sequence[str] = CAMERA_NAMES,
                 epoch: float = 0.0) -> None:
        self.dir = Path(frames_dir)
        self.names = tuple(names)
        self.epoch = epoch
        self.tick = 0

    def at(self, tick: int, max_age_s: float = DEFAULT_MAX_AGE_S) -> dict[str, Frame]:
        now = time.monotonic()
        out: dict[str, Frame] = {}
        for name in self.names:
            p = self.dir / f"{name}_{tick:06d}.png"
            if not p.exists():
                raise CameraError(f"{name}: no recorded frame at {p}")
            out[name] = Frame(name, p.read_bytes(), now, self.epoch, tick)
        return out

    def snapshot(self, max_age_s: float = DEFAULT_MAX_AGE_S) -> dict[str, Frame]:
        return self.at(self.tick, max_age_s)


def frames_to_packet_refs(frames: dict[str, Frame]) -> dict[str, Any]:
    """Shape `packet.build_packet` expects, from live frames."""
    return {name: {"frame_index": f.frame_id, "frame_present": True,
                   "grabbed_epoch": f.grabbed_epoch, "live": True}
            for name, f in frames.items()}


def frames_to_data_urls(frames: dict[str, Frame]) -> list[str]:
    import base64
    return ["data:image/png;base64," + base64.b64encode(f.png).decode()
            for _, f in sorted(frames.items()) if f.png]


def detect_cameras() -> list[int]:
    """Probe /dev/video*. Returns indices that actually open and report a size.

    Deliberately returns indices only, never a name guess: pairing the wrong
    device with 'base' or 'wrist' would feed the reviewer the wrong viewpoint
    while looking entirely normal.
    """
    import glob
    try:
        import cv2
    except ImportError:
        return []
    out = []
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if w > 0 and h > 0:
                out.append(idx)
    return out
