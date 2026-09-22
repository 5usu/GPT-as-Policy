"""Jetson launcher for hybrid_rollout.robodojo.kuka.pi05_serve (proposal-only).

The committed pi05_serve cannot load pi05_corrected_b8 on lerobot 0.5.1:
  - PreTrainedConfig.from_pretrained rejects the training-fork config key
    `pretrained_revision` (draccus DecodingError), and the relative-action
    pre/post steps need a registry alias + a hand-made pairing;
  - predict() passes a dict to a postprocessor that takes a tensor, and never
    scales uint8 images to [0, 1].
This wrapper swaps in the production-proven KUKA loader/inference path
(teleoperation/pi05_standalone.py) and keeps pi05_serve's HTTP surface unchanged.

Images in POST /infer: {"images": {"base": <b64 PNG/JPEG | HxWx3 uint8 list>, ...}}
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/KUKA/teleoperation")      # pi05_standalone
sys.path.insert(0, "/workspace/GPT-as-Policy")

import numpy as np
from hybrid_rollout.robodojo.kuka import pi05_serve


def _image(v) -> np.ndarray:
    if isinstance(v, str):
        from PIL import Image
        if v.startswith("data:"):
            v = v.split(",", 1)[1]
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB"))
    arr = np.asarray(v)
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr


def load_policy(checkpoint: str, device: str = "cuda"):
    from pi05_standalone import load_pi05_policy
    policy, pre, post, dev = load_pi05_policy(checkpoint, device)
    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    meta = {"type": raw.get("type"), "chunk_size": raw.get("chunk_size"),
            "n_action_steps": raw.get("n_action_steps"),
            "use_relative_actions": raw.get("use_relative_actions"),
            "relative_exclude_joints": raw.get("relative_exclude_joints"),
            "action_feature_names": raw.get("action_feature_names"),
            "image_features": sorted(k for k in raw.get("input_features", {})
                                     if k.startswith("observation.images.")),
            "checkpoint": checkpoint, "device": str(dev),
            "loader": "KUKA pi05_standalone.load_pi05_policy"}
    pi05_serve._STATE["device"] = dev
    return policy, pre, post, meta


def predict(observation: dict) -> dict:
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    policy, pre, post = (pi05_serve._STATE[k] for k in ("policy", "pre", "post"))
    if policy is None:
        return {"ok": False, "error": "no policy loaded"}
    try:
        dev = pi05_serve._STATE["device"]
        obs = {"observation.state": np.asarray(observation["state"], dtype=np.float32)}
        for k, v in (observation.get("images") or {}).items():
            obs[f"observation.images.{k}"] = _image(v)
        with torch.inference_mode(), torch.autocast(device_type=dev.type):
            batch = prepare_observation_for_inference(obs, dev, observation.get("task", ""))
            batch = pre(batch)
            chunk = policy.predict_action_chunk(batch)          # [1, T, D], relative + normalised
            chunk = post(chunk)                                 # absolute degrees + gripper
        rows = chunk[0].float().cpu().tolist()
        return {"ok": True, "rows": rows, "n_steps": len(rows),
                "action_space": "absolute_joint_targets_deg",
                "checkpoint_id": pi05_serve._STATE["meta"].get("checkpoint"),
                "images_used": sorted((observation.get("images") or {}).keys()),
                "meta": pi05_serve._STATE["meta"]}
    except Exception as exc:                                    # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}


pi05_serve.load_policy = load_policy
pi05_serve.predict = predict

if __name__ == "__main__":
    raise SystemExit(pi05_serve.main())
