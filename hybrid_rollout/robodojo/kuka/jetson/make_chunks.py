"""Run the local pi0.5 checkpoint over an episode's sampled ticks -> chunks JSON.

    python3 make_chunks.py <episode_dir> <out.json> [--every 50] [--limit 5]

This is the "run pi0.5 locally" half that `cli run --chunks-file` expects since
the package dropped its serving layer. It loads the checkpoint with the
production KUKA loader (pi05_standalone), which handles the relative-action
pairing the plain lerobot recipe gets wrong, and writes:

    {"<episode_id>:t<tick:06d>": [[A1..A6, gripper] x 50], ..., "_meta": {...}}

Observation ids and states come from RecordedEpisode itself, so they match what
the CLI samples tick for tick. Runs inside the pi05-infer image; no robot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/KUKA/teleoperation")      # pi05_standalone
sys.path.insert(0, "/workspace/GPT-as-Policy")

import numpy as np
import torch
from PIL import Image

from hybrid_rollout.robodojo.kuka.episode import RecordedEpisode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--checkpoint", default="/ckpt/pretrained_model")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--chunk-steps", type=int, default=50)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ep_dir = Path(a.episode)

    ep = RecordedEpisode.open(
        ep_dir / "manifest.json", media_root=ep_dir,
        state_rows=json.loads((ep_dir / "state.json").read_text()),
        traj_rows=json.loads((ep_dir / "trajectory.json").read_text()))
    samples = ep.sample_ticks(every=a.every, chunk_steps=a.chunk_steps, limit=a.limit)
    missing = [f"{s.observation_id}/{c}" for s in samples
               for c, f in s.frames.items() if not f.get("frame_present")]
    if missing:
        raise SystemExit(f"frames not extracted: {missing[:3]}")

    from lerobot.policies.utils import prepare_observation_for_inference
    from pi05_standalone import load_pi05_policy
    policy, pre, post, dev = load_pi05_policy(a.checkpoint, a.device)
    raw = json.loads((Path(a.checkpoint) / "config.json").read_text())

    out: dict = {"_meta": {
        "use_relative_actions": raw.get("use_relative_actions"),
        "relative_exclude_joints": raw.get("relative_exclude_joints"),
        "action_names": raw.get("action_feature_names"),
        "chunk_size": raw.get("chunk_size"),
        "checkpoint": a.checkpoint, "episode_id": ep.m.episode_id,
        "task": ep.m.instruction, "device": str(dev),
        "loader": "KUKA pi05_standalone.load_pi05_policy",
        "action_space": "absolute_joint_targets_deg"}}
    for s in samples:
        obs = {"observation.state": np.asarray(s.state, dtype=np.float32)}
        for cam, f in sorted(s.frames.items()):
            obs[f"observation.images.{cam}"] = np.asarray(
                Image.open(f["frame_path"]).convert("RGB"))
        with torch.inference_mode(), torch.autocast(device_type=dev.type):
            batch = pre(prepare_observation_for_inference(obs, dev, ep.m.instruction))
            rows = post(policy.predict_action_chunk(batch))[0].float().cpu().tolist()
        drift = max(abs(x - y) for x, y in zip(rows[0][:6], s.state[:6]))
        print(f"  {s.observation_id}  {len(rows)}x{len(rows[0])}  "
              f"|first-state|={drift:.2f} deg", flush=True)
        out[s.observation_id] = rows
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {len(out) - 1} chunk(s) -> {a.out}")


if __name__ == "__main__":
    main()
