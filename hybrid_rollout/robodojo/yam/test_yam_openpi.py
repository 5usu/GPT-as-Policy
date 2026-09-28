"""The reconstructed openpi config, without weights. Skipped without openpi.

Runs the exact transform chain `create_trained_policy` builds -- data inputs,
quantile Normalize, pi05 model inputs; model outputs, Unnormalize, data outputs
-- on a synthetic observation, with the checkpoint's own q01/q99 action range.
This is where a swapped camera key, a missing pad or the wrong norm mode would
show, and it needs no GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

openpi = pytest.importorskip("openpi")

from .conftest import MEAN_STATE                              # noqa: E402
from .contract import (CAMERA_NAMES, CHUNK_STEPS, DATA_ENVELOPE_Q01,  # noqa: E402
                       DATA_ENVELOPE_Q99, IMAGE_SOURCE_HW)
from .pi05_serve import OPENPI_CAMERA_KEYS, openpi_train_config  # noqa: E402


@pytest.fixture(scope="module")
def chain():
    from openpi import transforms as _t
    cfg = openpi_train_config()
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    q01, q99 = np.asarray(DATA_ENVELOPE_Q01), np.asarray(DATA_ENVELOPE_Q99)
    stats = {k: _t.NormStats(mean=(q01 + q99) / 2, std=(q99 - q01) / 4, q01=q01, q99=q99)
             for k in ("state", "actions")}
    inputs = _t.compose([*dc.data_transforms.inputs,
                         _t.Normalize(stats, use_quantiles=dc.use_quantile_norm),
                         *dc.model_transforms.inputs])
    outputs = _t.compose([*dc.model_transforms.outputs,
                          _t.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
                          *dc.data_transforms.outputs])
    return cfg, dc, inputs, outputs


def test_model_is_pi05_with_sixteen_step_chunks(chain):
    cfg, dc, _, _ = chain
    assert cfg.model.pi05 and cfg.model.action_horizon == CHUNK_STEPS
    assert cfg.model.discrete_state_input and dc.use_quantile_norm
    assert dc.asset_id == "yam-bimanual-merged"


def test_inputs(chain):
    _, _, inputs, _ = chain
    h, w = IMAGE_SOURCE_HW
    imgs = {c: np.full((h, w, 3), 40 * (i + 1), np.uint8) for i, c in enumerate(CAMERA_NAMES)}
    out = inputs({"images": imgs, "state": np.asarray(MEAN_STATE, np.float32),
                  "prompt": "stack the blocks"})
    assert set(out["image"]) == set(OPENPI_CAMERA_KEYS.values())
    for cam, key in OPENPI_CAMERA_KEYS.items():
        img = out["image"][key]
        assert img.shape == (224, 224, 3)
        # 16:9 padded to square: the top rows are letterbox, the middle is the frame
        assert img[0, 112].max() == 0 and img[112, 112].max() > 0
        assert bool(out["image_mask"][key])
    # distinct fill per camera proves the mapping, not just the keys
    means = [float(out["image"][OPENPI_CAMERA_KEYS[c]][112, 112, 0]) for c in CAMERA_NAMES]
    assert means == sorted(means) and len(set(means)) == 3
    assert out["state"].shape == (32,)
    assert out["tokenized_prompt"].shape == (200,)


def test_outputs_unnormalise_to_fourteen_absolute_targets(chain):
    _, _, _, outputs = chain
    out = outputs({"actions": np.zeros((CHUNK_STEPS, 32), np.float32),
                   "state": np.zeros(32, np.float32)})
    a = out["actions"]
    assert a.shape == (CHUNK_STEPS, 14)
    mid = (np.asarray(DATA_ENVELOPE_Q01) + np.asarray(DATA_ENVELOPE_Q99)) / 2
    np.testing.assert_allclose(a[0], mid, atol=1e-4)
