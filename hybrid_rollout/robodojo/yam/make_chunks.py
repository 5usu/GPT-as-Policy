"""Run robocurve/pi0.5-yam over a recorded episode -> chunks.json. NO ROBOT.

    python -m hybrid_rollout.robodojo.yam.make_chunks \
        --checkpoint /ckpt/pi0.5-yam --episode /data/yam_ep \
        --every 16 --limit 20 --out /data/yam_ep/chunks.json

The YAM counterpart of `kuka.make_chunks`, and the first thing to run: it is
the only offline check that the reconstructed openpi config in `pi05_serve`
reproduces the checkpoint. For every sampled tick it also reports the error
against the RECORDED actions over the same 16 steps, so a wiring mistake -- a
swapped camera, the wrong gripper polarity, a missing normalisation -- shows up
as a large error on a held-out recording rather than as a strange motion on
the arms.

It loads a GPU model, reads files and writes one JSON file. It imports no robot
code and opens no socket.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .contract import CAMERA_NAMES, CHUNK_STEPS, JOINT_INDICES, GRIPPER_INDICES


def chunk_error(pred, recorded) -> dict[str, float]:
    """RMS joint error (rad) and max gripper error over the overlapping steps."""
    n = min(len(pred), len(recorded))
    if n == 0:
        return {"n": 0}
    sq = [(pred[i][j] - recorded[i][j]) ** 2 for i in range(n) for j in JOINT_INDICES]
    grip = [abs(pred[i][g] - recorded[i][g]) for i in range(n)
            for g in GRIPPER_INDICES.values()]
    return {"n": n, "joint_rms_rad": round(math.sqrt(sum(sq) / len(sq)), 5),
            "gripper_max_abs": round(max(grip), 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=CHUNK_STEPS)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--revision")
    a = ap.parse_args(argv)

    import numpy as np
    from PIL import Image

    from .episode import YamRecordedEpisode
    from .pi05_serve import ContractViolation, checkpoint_meta, load_policy

    ep_dir = Path(a.episode)
    ep = YamRecordedEpisode.open(
        ep_dir / "manifest.json", media_root=ep_dir,
        state_rows=json.loads((ep_dir / "state.json").read_text()),
        traj_rows=json.loads((ep_dir / "trajectory.json").read_text()))
    samples = ep.sample_ticks(every=a.every, limit=a.limit)
    missing = [f"{s.observation_id}/{c}" for s in samples
               for c in CAMERA_NAMES if not (s.frames.get(c) or {}).get("frame_present")]
    if missing:
        print(f"REFUSED: frames not extracted for {missing[:3]} (and "
              f"{max(0, len(missing) - 3)} more). Expected cameras {CAMERA_NAMES}.")
        return 2
    try:
        meta = checkpoint_meta(a.checkpoint, a.revision)
    except ContractViolation as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"episode : {ep.m.episode_id}  task={ep.m.instruction!r}")
    print(f"loading : {a.checkpoint} (revision {meta['revision'][:12]})")
    policy = load_policy(a.checkpoint)

    out: dict[str, object] = {"_meta": {**meta, "episode_id": ep.m.episode_id,
                                        "task": ep.m.instruction}}
    errors = []
    for s in samples:
        images = {c: np.asarray(Image.open(s.frames[c]["frame_path"]).convert("RGB"))
                  for c in CAMERA_NAMES}
        res = policy.infer({"images": images, "prompt": ep.m.instruction,
                            "state": np.asarray(s.state, dtype=np.float32)})
        rows = np.asarray(res["actions"], dtype=np.float64).tolist()
        err = chunk_error(rows, s.recorded_chunk)
        errors.append(err)
        out[s.observation_id] = rows
        print(f"  {s.observation_id:28s} {len(rows)}x{len(rows[0])}  "
              f"joint rms {err.get('joint_rms_rad')} rad  "
              f"gripper max {err.get('gripper_max_abs')}")
    dst = Path(a.out or (ep_dir / "chunks.json"))
    out["_meta"]["open_loop"] = errors                       # type: ignore[index]
    dst.write_text(json.dumps(out))
    rms = [e["joint_rms_rad"] for e in errors if e.get("n")]
    if rms:
        print(f"\nmean joint rms vs recorded actions: {sum(rms) / len(rms):.5f} rad")
    print(f"wrote {len(samples)} chunk(s) -> {dst}")
    print("NO ROBOT TOUCHED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
