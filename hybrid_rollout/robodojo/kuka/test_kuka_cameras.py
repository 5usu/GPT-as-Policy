"""Live camera layer. No device is opened; fakes stand in for /dev/video*.

The property that matters is freshness: a stale frame looks like a valid
observation while describing a pose the arm has already left.
"""
from __future__ import annotations

import time

import pytest

from .cameras import (CAMERA_NAMES, DEFAULT_MAX_AGE_S, FPS, HEIGHT, WIDTH,
                      CameraError, Frame, LiveCameras, RecordedCameras,
                      detect_cameras, frames_to_data_urls, frames_to_packet_refs)


def frame(name, age_s=0.0, png=b"\x89PNG-fake"):
    now = time.monotonic()
    return Frame(name, png, now - age_s, time.time() - age_s, 1)


class _FakeGrabber:
    def __init__(self, f):
        self._f = f

    def latest(self):
        return self._f

    def stop(self):
        pass

    def to_log(self):
        return {"camera": getattr(self._f, "camera", "?")}


def live_with(ages: dict[str, float | None]) -> LiveCameras:
    c = LiveCameras(mapping={n: i for i, n in enumerate(ages)}, names=tuple(ages))
    c._grabbers = {n: _FakeGrabber(None if a is None else frame(n, a))
                   for n, a in ages.items()}
    return c


class TestMatchesDeployedCapture:
    def test_geometry_matches_training(self):
        assert (WIDTH, HEIGHT, FPS) == (640, 480, 30)

    def test_camera_names_match_the_dataset(self):
        assert CAMERA_NAMES == ("base", "wrist")


class TestFreshness:
    def test_fresh_frames_returned(self):
        s = live_with({"base": 0.01, "wrist": 0.02}).snapshot()
        assert set(s) == {"base", "wrist"}

    def test_one_stale_camera_refuses_the_whole_snapshot(self):
        """A half-fresh observation is not an observation."""
        with pytest.raises(CameraError) as e:
            live_with({"base": 0.01, "wrist": 5.0}).snapshot()
        assert "wrist" in str(e.value) and "old" in str(e.value)

    def test_missing_frame_is_an_error_not_an_absent_key(self):
        with pytest.raises(CameraError) as e:
            live_with({"base": 0.01, "wrist": None}).snapshot()
        assert "no frame yet" in str(e.value)

    def test_max_age_is_enforced_not_advisory(self):
        c = live_with({"base": 0.30})
        with pytest.raises(CameraError):
            c.snapshot(max_age_s=0.25)
        assert c.snapshot(max_age_s=1.0)

    def test_default_bound_is_about_seven_frames(self):
        assert DEFAULT_MAX_AGE_S == pytest.approx(7 / 30, abs=0.03)

    def test_age_is_monotonic_based(self):
        f = frame("base", 0.5)
        assert f.age_s() == pytest.approx(0.5, abs=0.05)


class TestNoGuessing:
    def test_start_refuses_without_an_explicit_mapping(self):
        """Device order is not stable; guessing pairs the wrong view with a name."""
        with pytest.raises(CameraError) as e:
            LiveCameras().start()
        assert "not stable across" in str(e.value)

    def test_partial_mapping_refused(self):
        with pytest.raises(CameraError):
            LiveCameras(mapping={"base": 0}).start()

    def test_detect_returns_indices_only(self):
        """It must not suggest names -- that is the pairing it refuses to guess."""
        out = detect_cameras()
        assert isinstance(out, list)
        assert all(isinstance(i, int) for i in out)

    def test_nothing_opens_on_construction(self):
        c = LiveCameras(mapping={"base": 0, "wrist": 1})
        assert c._grabbers == {}


class TestPacketInterop:
    def test_refs_have_the_shape_build_packet_wants(self):
        refs = frames_to_packet_refs({"base": frame("base"), "wrist": frame("wrist")})
        for r in refs.values():
            assert r["frame_present"] is True and r["live"] is True
            assert "frame_index" in r and "grabbed_epoch" in r

    def test_data_urls_are_sorted_and_png(self):
        urls = frames_to_data_urls({"wrist": frame("wrist"), "base": frame("base")})
        assert len(urls) == 2
        assert all(u.startswith("data:image/png;base64,") for u in urls)

    def test_build_packet_accepts_live_refs(self):
        from .packet import build_packet
        refs = frames_to_packet_refs({"base": frame("base"), "wrist": frame("wrist")})
        p = build_packet(task_instruction="open the white dishwasher on the table",
                         observation_id="live:0", state=[0.0] * 7,
                         chunk=[[0.0] * 7] * 50, provenance="model_predicted",
                         frames=refs)
        assert "base" in p["user_text"] and "wrist" in p["user_text"]


class TestRecordedCameras:
    def test_reads_extracted_frames(self, tmp_path):
        for cam in CAMERA_NAMES:
            (tmp_path / f"{cam}_000080.png").write_bytes(b"\x89PNG")
        s = RecordedCameras(tmp_path).at(80)
        assert set(s) == set(CAMERA_NAMES)

    def test_missing_recorded_frame_errors(self, tmp_path):
        with pytest.raises(CameraError):
            RecordedCameras(tmp_path).at(999)

    def test_same_interface_as_live(self):
        assert hasattr(RecordedCameras("/tmp"), "snapshot")
        assert hasattr(LiveCameras(), "snapshot")
