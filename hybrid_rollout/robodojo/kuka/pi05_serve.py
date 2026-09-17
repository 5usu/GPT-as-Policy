"""pi0.5 inference server. RUNS ON THE A800 ONLY. Proposal-only.

    python -m hybrid_rollout.robodojo.kuka.pi05_serve \
        --checkpoint /path/to/pi05_corrected_b8/132000/pretrained_model --port 8710

This is the ONLY module that loads a GPU model, and it is deliberately separate
from everything else: it has no import path to the gateway, no robot code, and
no knowledge that a KUKA exists beyond the action dimensions. It answers
"what would the policy do here" and nothing else.

THE PROCESSOR RECIPE IS NOT OPTIONAL
The checkpoint was trained with use_relative_actions=True, which is an INTERNAL
transform: lerobot subtracts the current state on input and adds it back on
output. Getting that wrong does not crash - it silently returns joint targets in
the wrong space, which is exactly the failure that produced "no meaningful
actions" on the robot. So:

  - pre/post processors MUST come from `make_pre_post_processors`, because
    `PolicyProcessorPipeline.from_pretrained` leaves the unnormalizer unpaired
  - the postprocessed chunk is ABSOLUTE joint targets in degrees
  - HF must be offline; a network fetch mid-run changes the tokenizer underneath
    the model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SCHEMA = "hybrid_rollout.robodojo.kuka.pi05_serve.v1"
_STATE = {"policy": None, "pre": None, "post": None, "meta": {}}


def load_policy(checkpoint: str, device: str = "cuda"):
    """Load with the verified recipe. Offline, paired processors."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint)
    meta = {"type": cfg.type, "chunk_size": getattr(cfg, "chunk_size", None),
            "n_action_steps": getattr(cfg, "n_action_steps", None),
            "use_relative_actions": getattr(cfg, "use_relative_actions", None),
            "action_feature_names": getattr(cfg, "action_feature_names", None),
            "checkpoint": checkpoint, "device": device}
    if meta["use_relative_actions"] is not True:
        print("WARNING: use_relative_actions is not True. This is very likely the "
              "WRONG checkpoint -- the models/ directory in OSS holds finetunes "
              "with it set false.", file=sys.stderr)
    return policy, pre, post, meta


def predict(observation: dict) -> dict:
    """observation -> {rows: 50x7 ABSOLUTE joint targets deg}."""
    import numpy as np
    import torch
    policy, pre, post = _STATE["policy"], _STATE["pre"], _STATE["post"]
    if policy is None:
        return {"ok": False, "error": "no policy loaded"}
    try:
        batch = {"observation.state": torch.tensor(
            [observation["state"]], dtype=torch.float32)}
        for k, v in (observation.get("images") or {}).items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[None]
            batch[f"observation.images.{k}"] = torch.tensor(arr)
        batch["task"] = [observation.get("task", "")]
        dev = next(policy.parameters()).device
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            processed = pre(batch)
            out = policy.predict_action_chunk(processed)
            out = post({"action": out})
        rows = out["action"][0].detach().cpu().tolist()
        return {"ok": True, "rows": rows, "n_steps": len(rows),
                "action_space": "absolute_joint_targets_deg",
                "checkpoint_id": _STATE["meta"].get("checkpoint"),
                "meta": _STATE["meta"]}
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                          # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": _STATE["policy"] is not None,
                             "schema": SCHEMA, "meta": _STATE["meta"]})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):                                         # noqa: N802
        if self.path != "/infer":
            return self._json(404, {"ok": False, "error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            obs = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad json"})
        self._json(200, predict(obs))

    def log_message(self, *a):                                 # quieter
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8710)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    print(f"loading {a.checkpoint} on {a.device} ...")
    _STATE["policy"], _STATE["pre"], _STATE["post"], _STATE["meta"] = \
        load_policy(a.checkpoint, a.device)
    print("loaded:", json.dumps(_STATE["meta"], indent=1))
    print(f"serving on http://{a.host}:{a.port}  (POST /infer, GET /health)")
    print("PROPOSAL-ONLY. This process has no robot connection.")
    HTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
