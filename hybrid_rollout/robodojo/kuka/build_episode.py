"""Build a reviewable episode from a LeRobot v3 dataset. Offline except OSS reads.

    python -m hybrid_rollout.robodojo.kuka.build_episode \
        --dataset /path/merge_all_factory_clean_v1_val_ood \
        --task-match "open the white dishwasher" --out /root/kuka_ep

Produces the layout `episode.RecordedEpisode` expects: manifest.json, state.json,
trajectory.json, media/{base,wrist}.mp4 and frames/.

WHY THIS IS NOT TRIVIAL
The val merges are VIRTUAL: `meta/_video_source_map.parquet` points each episode's
videos at an OSS prefix rather than a local file, so the videos have to be
fetched before anything can be reviewed. They are small -- a per-episode file is
a couple of MB, not gigabytes -- which is worth knowing because it makes running
this over many episodes cheap.

Timestamps are SYNTHESISED from the dataset's own frame rate, with the video and
state streams sharing one origin. That is honest for a recorded dataset where
both streams came from the same capture, and it is what makes the manifest's
alignment check meaningful rather than vacuous. A live capture must record real
wall-clock instead.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CAMERAS = ("base", "wrist")
DEFAULT_ORIGIN = 1_000_000.0


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_oss_credentials() -> dict[str, str]:
    """Read credentials from the operator's files. Values are never logged."""
    out: dict[str, str] = {}
    for p in ("/root/.oss_credentials", "/root/KUKA/teleoperation/.env"):
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return out


def pick_episode(dataset: str, task_match: str | None):
    import pyarrow.parquet as pq
    tp = os.path.join(dataset, "meta", "tasks.parquet")
    tasks = {}
    if os.path.exists(tp):
        t = pq.read_table(tp).to_pydict()
        keys = list(t)
        if len(keys) >= 2:
            tasks = {int(i): str(s) for i, s in zip(t[keys[0]], t[keys[1]])}
    wanted = None
    if task_match:
        wanted = {i for i, s in tasks.items() if task_match.lower() in s.lower()}
        if not wanted:
            raise SystemExit(f"no task matching {task_match!r}; have {list(tasks.values())}")
    for f in sorted(glob.glob(os.path.join(dataset, "data", "*", "*.parquet"))):
        d = pq.read_table(f).to_pydict()
        ti = d.get("task_index") or []
        if not ti:
            continue
        tid = int(ti[0])
        if wanted is None or tid in wanted:
            return int(d["episode_index"][0]), tid, tasks.get(tid, ""), d
    raise SystemExit("no matching episode found")


def fetch_videos(dataset: str, episode: int, out: Path) -> dict[str, Path]:
    """Resolve the virtual video map and pull each camera's file from OSS."""
    import oss2
    import pyarrow.parquet as pq
    smap = os.path.join(dataset, "meta", "_video_source_map.parquet")
    if not os.path.exists(smap):
        raise SystemExit(f"no video source map at {smap}")
    t = pq.read_table(smap).to_pydict()
    rows = [(t["video_key"][i], t["src_prefix"][i], t["src_relpath"][i])
            for i in range(len(t["episode_index"]))
            if int(t["episode_index"][i]) == episode]
    if not rows:
        raise SystemExit(f"no video sources for episode {episode}")
    c = load_oss_credentials()
    auth = oss2.Auth(c.get("OSS_ACCESS_KEY_ID"), c.get("OSS_ACCESS_KEY_SECRET"))
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    got: dict[str, Path] = {}
    for bucket, endpoint in (("i-robot-data", "https://oss-cn-guangzhou.aliyuncs.com"),
                             ("intuition-video-lrs", "https://oss-cn-shanghai.aliyuncs.com")):
        b = oss2.Bucket(auth, endpoint, bucket)
        try:
            for key, pre, rel in rows:
                b.head_object(f"{pre}/{rel}")
        except Exception:
            continue
        for key, pre, rel in rows:
            cam = key.split(".")[-1]
            dst = media / f"{cam}.mp4"
            b.get_object_to_file(f"{pre}/{rel}", str(dst))
            got[cam] = dst
            print(f"  {cam}: {dst.stat().st_size/1e6:.1f} MB from {bucket}")
        return got
    raise SystemExit("videos not found in any readable bucket")


def extract_frames(media: dict[str, Path], ticks, fps: float, out: Path) -> int:
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    n = 0
    for cam, src in media.items():
        for t in ticks:
            dst = frames / f"{cam}_{t:06d}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t / fps),
                            "-i", str(src), "-frames:v", "1", str(dst)], check=False)
            n += dst.exists()
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--task-match")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--checkpoint-id", default="pi05_corrected_b8/132000")
    p.add_argument("--checkpoint-sha256",
                   default="c881c42551bfc173064f2c9f33a1ff7c61a7dfc1ad49172dc1d6c384111d72d0")
    p.add_argument("--no-videos", action="store_true")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ep, tid, task, d = pick_episode(a.dataset, a.task_match)
    n = len(d["frame_index"])
    print(f"episode ep{ep:06d}  task_index={tid}  task={task!r}  frames={n}")

    (out / "state.json").write_text(json.dumps(
        [[float(v) for v in r] for r in d["observation.state"]]))
    (out / "trajectory.json").write_text(json.dumps(
        [[float(v) for v in r] for r in d["action"]]))

    media = {} if a.no_videos else fetch_videos(a.dataset, ep, out)
    ticks = list(range(0, max(1, n - 50), a.every))[:a.limit]
    got = extract_frames(media, ticks, a.fps, out) if media else 0
    print(f"extracted {got} frame(s) at ticks {ticks}")

    origin = DEFAULT_ORIGIN
    man = {
        "schema": "hybrid_rollout.robodojo.kuka.episode_manifest.v1",
        "episode_id": f"ep{ep:06d}",
        "dataset": os.path.basename(a.dataset.rstrip("/")),
        "task": {"name": "dishwasher_door", "instruction": task},
        "checkpoint": {"id": a.checkpoint_id, "sha256": a.checkpoint_sha256,
                       "action_space": "absolute_joint_targets_deg",
                       "action_names": ["A1", "A2", "A3", "A4", "A5", "A6", "gripper"]},
        "timestamps": {"recorded_start_epoch": origin,
                       "recorded_end_epoch": origin + n / a.fps,
                       "control_hz": a.fps, "timezone": "UTC"},
        "videos": [{"camera": cam, "path": f"media/{cam}.mp4",
                    "sha256": sha256(media[cam]), "n_frames": n, "fps": a.fps,
                    "width": 640, "height": 480, "first_frame_epoch": origin}
                   for cam in CAMERAS if cam in media],
        "state": {"path": "state.json", "sha256": sha256(out / "state.json"),
                  "n_rows": n,
                  "columns": ["A1", "A2", "A3", "A4", "A5", "A6", "gripper"],
                  "first_row_epoch": origin},
        "trajectory": {"path": "trajectory.json",
                       "sha256": sha256(out / "trajectory.json"), "n_rows": n,
                       "provenance": "recorded_demo"},
        "calibration": {"intrinsics": None, "extrinsics": None, "verified": False},
        "outcome": {"success": None, "source": "unknown", "predicate_value": None,
                    "notes": "recorded demonstration; outcome not measured"},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
