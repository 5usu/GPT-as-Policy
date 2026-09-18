"""Run pi0.5 locally and write chunks.json for the review loop. NO ROBOT.

    python -m hybrid_rollout.robodojo.kuka.make_chunks \
        --checkpoint /path/to/132000/pretrained_model \
        --episode /path/to/kuka_ep --every 50 --limit 6 \
        --out /path/to/kuka_ep/chunks.json

This is the ONLY module that loads a GPU model, and it does nothing else: it
reads recorded observations, runs the policy, and writes a JSON file. It opens no
socket, imports no robot code, and cannot command anything.

THE PROCESSOR RECIPE IS NOT OPTIONAL
The checkpoint trains with use_relative_actions=True, an INTERNAL transform:
lerobot subtracts the current state on input and adds it back on output. Get it
wrong and nothing crashes -- you silently get joint targets in the wrong space,
which is the shape of the "no meaningful actions" failure seen on the robot. So:

  - pre/post processors MUST come from `make_pre_post_processors`, because
    `PolicyProcessorPipeline.from_pretrained` leaves the unnormalizer unpaired
  - HF must be offline; a mid-run fetch can change the tokenizer underneath
  - the postprocessed chunk is ABSOLUTE joint targets in degrees

The written `_meta.use_relative_actions` is what `LocalPi05ProposalSource` checks
before it will accept the file, so a wrong-checkpoint run is refused downstream
rather than silently reviewed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_policy(checkpoint: str, device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint)
    meta = {"checkpoint": str(checkpoint), "type": cfg.type,
            "chunk_size": getattr(cfg, "chunk_size", None),
            "n_action_steps": getattr(cfg, "n_action_steps", None),
            "use_relative_actions": getattr(cfg, "use_relative_actions", None),
            "action_feature_names": getattr(cfg, "action_feature_names", None),
            "device": device}
    return policy, pre, post, meta


def frame_array(path: Path):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode", required=True,
                    help="directory with manifest.json, state.json, frames/")
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-images", action="store_true",
                    help="state-only, for a quick load check")
    a = ap.parse_args(argv)

    ep = Path(a.episode)
    man = json.loads((ep / "manifest.json").read_text())
    state_rows = json.loads((ep / "state.json").read_text())
    task = (man.get("task") or {}).get("instruction", "")
    hz = float((man.get("timestamps") or {}).get("control_hz") or 30.0)
    cams = [v["camera"] for v in (man.get("videos") or [])]

    print(f"episode   : {man.get('episode_id')}  ticks={len(state_rows)} @ {hz:.0f} Hz")
    print(f"task      : {task!r}")
    print(f"loading   : {a.checkpoint} on {a.device}")
    policy, pre, post, meta = load_policy(a.checkpoint, a.device)
    print(f"  type                 : {meta['type']}")
    print(f"  chunk / n_action     : {meta['chunk_size']} / {meta['n_action_steps']}")
    print(f"  use_relative_actions : {meta['use_relative_actions']}", end="")
    if meta["use_relative_actions"] is not True:
        print("   <-- WRONG CHECKPOINT")
        print("\nRefusing: this build expects use_relative_actions=True. The eight "
              "finetunes under oss://i-robot-data/models/ are a different "
              "training run and will silently produce targets in the wrong "
              "space.", file=sys.stderr)
        return 2
    print("   OK")

    import torch
    ticks = list(range(0, max(1, len(state_rows) - 50), a.every))[:a.limit]
    out: dict[str, object] = {"_meta": meta}
    for t in ticks:
        batch = {"observation.state": torch.tensor([state_rows[t]],
                                                   dtype=torch.float32)}
        if not a.no_images:
            import numpy as np
            for cam in cams:
                p = ep / "frames" / f"{cam}_{t:06d}.png"
                if not p.exists():
                    print(f"  tick {t}: missing {p}; skipping", file=sys.stderr)
                    batch = None
                    break
                arr = frame_array(p).transpose(2, 0, 1)[None]
                batch[f"observation.images.{cam}"] = torch.tensor(arr)
        if batch is None:
            continue
        batch["task"] = [task]
        dev = next(policy.parameters()).device
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            chunk = post({"action": policy.predict_action_chunk(pre(batch))})
        rows = chunk["action"][0].detach().cpu().tolist()
        oid = f"{man.get('episode_id')}:t{t:06d}"
        out[oid] = rows
        first = [round(v, 3) for v in rows[0][:6]]
        print(f"  tick {t:6d} -> {oid:26s} {len(rows)}x{len(rows[0])}  first {first}")

    dst = Path(a.out or (ep / "chunks.json"))
    dst.write_text(json.dumps(out))
    n = len([k for k in out if not k.startswith("_")])
    print(f"\nwrote {n} chunk(s) -> {dst}")
    print("NO ROBOT TOUCHED. This produced a file and nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
