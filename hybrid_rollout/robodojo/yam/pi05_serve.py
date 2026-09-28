"""Serve robocurve/pi0.5-yam over HTTP. Proposal-only; holds no robot handle.

    # on the GPU host, in an openpi environment (JAX + CUDA)
    hf download robocurve/pi0.5-yam --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 \
        --local-dir /ckpt/pi0.5-yam
    python -m hybrid_rollout.robodojo.yam.pi05_serve --checkpoint /ckpt/pi0.5-yam

    GET  /health  -> {ok, meta}
    POST /infer   {state: [14], images: {top, left, right}, task: str}
               -> {ok, rows: 16x14, meta, latency_s}

WHY THIS FILE BUILDS ITS OWN OPENPI CONFIG
The model card loads the checkpoint with `oc.get_config("yam_pi05")`, a config
registered by robocurve's training repo, which is private. Rather than guess at
it, the config here is assembled from what is published and checkable:

  model       Pi0Config(pi05=True, action_horizon=16) -- 16-step chunks per the
              card; action_dim stays openpi's padded 32 and the output keeps 14
  norm        quantile (openpi's default for pi05), loaded from the checkpoint's
              OWN assets/yam-bimanual-merged/norm_stats.json, digest-checked
  images      top -> base_0_rgb, left -> left_wrist_0_rgb,
              right -> right_wrist_0_rgb; openpi's ResizeImages(224, 224) pads
              to preserve aspect, so send 360x640 frames as trained
  state       14 absolute values, gripper = last COMMANDED opening
  actions     absolute joint targets (no delta transform); the published
              Jetson Thor bundle serves this checkpoint the same way
              (vla-edge PolicySpec pi05-bimanual-yam: 16x14 absolute,
              closed_0_open_1, gripper_state=commanded)

This is a reconstruction, not robocurve's code. Before trusting it on the rig,
replay a held-out recording through it (`yam.make_chunks` over
allenai/19012026-block-13) and compare chunks with the recorded actions; the
card reports 0.00206 open-loop MSE in normalised space for its own loader.

THE CONTRACT TRAVELS WITH EVERY RESPONSE. The client (`transports.
check_checkpoint_meta`) refuses a chunk whose meta does not match, so a
different fine-tune behind the same port is caught at the first request.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .contract import (ACTION_DIM, CAMERA_NAMES, CHECKPOINT, CHUNK_STEPS,
                       GRIPPER_CONVENTION)

SCHEMA = "yam.pi05_serve.v1"
NORM_STATS_REL = Path("assets") / CHECKPOINT["norm_asset_id"] / "norm_stats.json"
OPENPI_CAMERA_KEYS = {"top": "base_0_rgb", "left": "left_wrist_0_rgb",
                      "right": "right_wrist_0_rgb"}


class ContractViolation(RuntimeError):
    """The checkpoint on disk is not the one this contract describes."""


def checkpoint_meta(checkpoint_dir: str | Path, revision: str | None = None) -> dict[str, Any]:
    """Identify the checkpoint from its files. Raises on a digest mismatch."""
    ckpt = Path(checkpoint_dir)
    stats = ckpt / NORM_STATS_REL
    if not stats.exists():
        raise ContractViolation(f"{stats} is missing; this is not a "
                                f"{CHECKPOINT['repo_id']} checkpoint directory")
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if digest != CHECKPOINT["norm_stats_sha256"]:
        raise ContractViolation(
            f"norm_stats.json sha256 {digest[:16]}... does not match the "
            f"contracted {CHECKPOINT['norm_stats_sha256'][:16]}...; a different "
            f"normalisation would unnormalise every action to the wrong place")
    if revision is None:
        m = re.search(r"snapshots/([0-9a-f]{40})", str(ckpt.resolve()))
        revision = m.group(1) if m else None
    return {"repo_id": CHECKPOINT["repo_id"], "revision": revision or "unknown",
            "revision_matches": revision == CHECKPOINT["revision"],
            "norm_stats_sha256": digest,
            "action_horizon": CHUNK_STEPS, "action_dim": ACTION_DIM,
            "action_space": CHECKPOINT["action_space"],
            "gripper_convention": GRIPPER_CONVENTION,
            "cameras": list(CAMERA_NAMES), "checkpoint": str(ckpt),
            "loader": "yam.pi05_serve openpi reconstruction",
            "openpi_commit_expected": CHECKPOINT["openpi_commit"]}


def openpi_train_config():
    """The reconstructed openpi TrainConfig. Imports openpi lazily."""
    import dataclasses

    import numpy as np
    from openpi import transforms as _t
    from openpi.models import pi0_config
    from openpi.training import config as oc

    @dataclasses.dataclass(frozen=True)
    class YamInputs(_t.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            imgs = data["images"]
            missing = [c for c in CAMERA_NAMES if c not in imgs]
            if missing:
                raise ValueError(f"missing cameras {missing}; expected {CAMERA_NAMES}")
            out = {"image": {OPENPI_CAMERA_KEYS[c]: np.asarray(imgs[c], dtype=np.uint8)
                             for c in CAMERA_NAMES},
                   "image_mask": {OPENPI_CAMERA_KEYS[c]: np.True_ for c in CAMERA_NAMES},
                   "state": np.asarray(data["state"], dtype=np.float32)}
            if "actions" in data:
                out["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if "prompt" in data:
                out["prompt"] = data["prompt"]
            return out

    @dataclasses.dataclass(frozen=True)
    class YamOutputs(_t.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}

    @dataclasses.dataclass(frozen=True)
    class YamDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=_t.Group(inputs=[YamInputs()], outputs=[YamOutputs()]),
                model_transforms=oc.ModelTransformFactory()(model_config))

    return oc.TrainConfig(
        name="yam_pi05_reconstructed",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=CHUNK_STEPS),
        data=YamDataConfig(assets=oc.AssetsConfig(asset_id=CHECKPOINT["norm_asset_id"])))


def load_policy(checkpoint_dir: str | Path):
    from openpi.policies import policy_config
    return policy_config.create_trained_policy(openpi_train_config(), str(checkpoint_dir))


def decode_image(value: Any):
    """base64 / data-URL JPEG or PNG, or an HxWx3 list -> HxWx3 uint8."""
    import numpy as np
    if isinstance(value, str):
        from PIL import Image
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {arr.shape}")
    return arr.astype(np.uint8)


class Server:
    """Holds one policy. Inference is serialised: one request at a time."""

    def __init__(self, policy, meta: dict[str, Any]) -> None:
        self.policy = policy
        self.meta = meta
        self._lock = threading.Lock()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        state = obs.get("state")
        if not isinstance(state, list) or len(state) != ACTION_DIM:
            return {"ok": False, "error": f"state must be {ACTION_DIM} values"}
        imgs = obs.get("images") or {}
        missing = [c for c in CAMERA_NAMES if c not in imgs]
        if missing:
            return {"ok": False,
                    "error": f"missing cameras {missing}; pi0.5 would infer from "
                             f"a black frame in their place"}
        task = str(obs.get("task") or "").strip()
        if not task:
            return {"ok": False, "error": "task (the language prompt) is required"}
        try:
            batch = {"images": {c: decode_image(imgs[c]) for c in CAMERA_NAMES},
                     "state": np.asarray(state, dtype=np.float32), "prompt": task}
            t0 = time.monotonic()
            with self._lock:
                out = self.policy.infer(batch)
            latency = time.monotonic() - t0
            rows = np.asarray(out["actions"], dtype=np.float64).tolist()
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        if len(rows) != CHUNK_STEPS or any(len(r) != ACTION_DIM for r in rows):
            return {"ok": False, "error": f"policy returned {len(rows)} rows, "
                                          f"expected {CHUNK_STEPS}x{ACTION_DIM}"}
        return {"ok": True, "rows": rows, "latency_s": round(latency, 4),
                "image_shapes": {c: list(batch["images"][c].shape) for c in CAMERA_NAMES}}


def make_handler(server: Server):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                   # noqa: N802
            if self.path.rstrip("/") in ("/health", "/healthz"):
                return self._send(200, {"ok": True, "schema": SCHEMA,
                                        "meta": server.meta,
                                        "note": "proposal-only; no robot handle"})
            return self._send(404, {"ok": False, "error": "GET /health only"})

        def do_POST(self):                                  # noqa: N802
            if self.path.rstrip("/") != "/infer":
                return self._send(404, {"ok": False, "error": "POST /infer only"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                obs = json.loads(self.rfile.read(n).decode())
            except Exception as exc:                        # noqa: BLE001
                return self._send(400, {"ok": False, "error": f"bad request: {exc}"[:200]})
            out = server.infer(obs if isinstance(obs, dict) else {})
            out["meta"] = server.meta
            out["schema"] = SCHEMA
            return self._send(200 if out.get("ok") else 422, out)

        def log_message(self, fmt, *a):                     # quieter
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    import argparse
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description="Serve robocurve/pi0.5-yam proposals.")
    p.add_argument("--checkpoint", required=True,
                   help="local directory holding params/ and assets/")
    p.add_argument("--revision", help="HF revision the directory was downloaded at")
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback by default; this serves a model, not a robot")
    p.add_argument("--port", type=int, default=18840)
    p.add_argument("--allow-revision-mismatch", action="store_true")
    a = p.parse_args(argv)

    try:
        meta = checkpoint_meta(a.checkpoint, a.revision)
    except ContractViolation as exc:
        print(json.dumps({"event": "refused", "error": str(exc)}), flush=True)
        return 2
    if not meta["revision_matches"] and not a.allow_revision_mismatch:
        print(json.dumps({"event": "refused", "error": (
            f"revision {meta['revision']!r} is not the contracted "
            f"{CHECKPOINT['revision']}; pass --revision if you downloaded "
            f"that one, or --allow-revision-mismatch")}), flush=True)
        return 2
    print(json.dumps({"event": "loading", "checkpoint": a.checkpoint}), flush=True)
    server = Server(load_policy(a.checkpoint), meta)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(server))
    print(json.dumps({"event": "loaded", "schema": SCHEMA, "host": a.host,
                      "port": a.port, "meta": meta}), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
