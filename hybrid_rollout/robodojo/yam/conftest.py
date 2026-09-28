"""Shared offline fixtures for the YAM tests. No robot, model, GPU or network."""
from __future__ import annotations

import json

import pytest

from .contract import ACTION_DIM, CAMERA_NAMES, CHECKPOINT, CHUNK_STEPS
from .experiment import MANIFEST_SCHEMA_ID

#: The checkpoint's mean training state -- a pose well inside every limit.
MEAN_STATE = [-0.1131, 1.4523, 1.2303, -0.6466, 0.3211, -0.3707, 0.6377,
              0.2210, 1.2591, 1.0209, -0.5300, -0.2051, 0.3185, 0.6242]


def chunk(state=None, *, step=0.01, n=CHUNK_STEPS, grip=None):
    """A smooth, feasible chunk: every arm joint advances `step` rad per row."""
    s = list(state or MEAN_STATE)
    rows = []
    for i in range(1, n + 1):
        r = list(s)
        for j in (*range(0, 6), *range(7, 13)):
            r[j] = s[j] + step * i
        if grip is not None:
            r[6], r[13] = grip
        rows.append(r)
    return rows


def good_meta(**over):
    m = {k: CHECKPOINT[k] for k in ("repo_id", "revision", "norm_stats_sha256",
                                    "action_horizon", "action_dim", "action_space",
                                    "gripper_convention")}
    m.update(over)
    return m


def decision(mode="student", steps=5, **extra):
    d = {"request_id": "x", "mode": mode, "steps": steps, "reason": "looks fine",
         "edit": {"left": {"delta_joint_rad": [0.0] * 6, "gripper": "keep"},
                  "right": {"delta_joint_rad": [0.0] * 6, "gripper": "keep"}},
         "assessment": {}}
    d.update(extra)
    return d


@pytest.fixture
def episode_dir(tmp_path):
    """A 64-tick recorded episode with tiny JPEG frames for every camera."""
    from PIL import Image
    n = 64
    state = [[v + 0.002 * t if j not in (6, 13) else v
              for j, v in enumerate(MEAN_STATE)] for t in range(n)]
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "trajectory.json").write_text(json.dumps(state[1:] + [state[-1]]))
    frames = tmp_path / "frames"
    frames.mkdir()
    for cam_i, cam in enumerate(CAMERA_NAMES):
        for t in range(n):
            Image.new("RGB", (64, 36), (40 * cam_i, t % 255, 90)).save(
                frames / f"{cam}_{t:06d}.jpg")
    manifest = {
        "schema": MANIFEST_SCHEMA_ID, "episode_id": "yam_test_ep",
        "task": {"instruction": "stack the blocks"},
        "checkpoint": {"id": None},
        "timestamps": {"control_hz": 30.0, "recorded_start_epoch": 0.0},
        "videos": [{"camera": c, "path": f"{c}.mp4", "sha256": "0" * 64, "fps": 30.0,
                    "n_frames": n, "first_frame_epoch": 0.0} for c in CAMERA_NAMES],
        "state": {"path": "state.json", "sha256": "1" * 64, "n_rows": n,
                  "first_row_epoch": 0.0},
        "trajectory": {"path": "trajectory.json", "sha256": "2" * 64,
                       "provenance": "recorded_demo"},
        "calibration": {}, "outcome": {"success": None, "source": "test"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


assert len(MEAN_STATE) == ACTION_DIM


@pytest.fixture(autouse=True)
def _no_rig_local(tmp_path, monkeypatch):
    """Tests never read the rig's own rig.local.toml."""
    monkeypatch.setenv("YAM_RIG_LOCAL", str(tmp_path / "no-rig.local.toml"))
