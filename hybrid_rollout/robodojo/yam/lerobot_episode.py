"""Convert one episode of a LeRobot v3 bimanual-YAM dataset into an episode dir.

    hf download allenai/19012026-block-13 --repo-type dataset --local-dir /data/block13
    python -m hybrid_rollout.robodojo.yam.lerobot_episode \
        --dataset /data/block13 --episode 0 --every 16 --limit 20 --out /data/yam_ep0

`allenai/19012026-block-13` is the session robocurve held out when selecting the
checkpoint, so it is the right recording to replay through the reconstructed
pi0.5 server (`yam.make_chunks`) before any arm is involved. Its feature layout
is the checkpoint's: 14-value state/action in contract order, cameras top/left/
right at 360x640, 30 fps.

Writes the layout `yam.episode` reads: manifest.json, state.json,
trajectory.json and frames/<camera>_<tick:06d>.jpg -- frames are extracted ONLY
at the ticks that will be sampled, with ffmpeg (AV1 needs an ffmpeg built with
libdav1d or libaom). Media stay outside this repository.

Needs pyarrow (`pip install pyarrow`) and ffmpeg. Neither is required anywhere
else in this package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .contract import ACTION_DIM, ACTION_NAMES, CAMERA_NAMES, CHUNK_STEPS
from .experiment import MANIFEST_SCHEMA_ID


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _episode_row(ds: Path, episode: int) -> dict:
    import pyarrow.parquet as pq
    for p in sorted((ds / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(p)
        idx = t.column("episode_index").to_pylist()
        if episode in idx:
            i = idx.index(episode)
            return {k: t.column(k)[i].as_py() for k in t.schema.names
                    if not k.startswith("stats/")}
    raise SystemExit(f"episode {episode} not found under {ds}/meta/episodes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="local LeRobot v3 dataset root")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=CHUNK_STEPS)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    ds, out = Path(a.dataset), Path(a.out)
    info = json.loads((ds / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    if len(names) != ACTION_DIM or info["features"]["observation.state"]["names"] != names:
        raise SystemExit(f"dataset layout {names} is not the 14-value contract layout")
    fps = float(info["fps"])
    row = _episode_row(ds, a.episode)
    data = ds / info["data_path"].format(chunk_index=row["data/chunk_index"],
                                         file_index=row["data/file_index"])
    t = pq.read_table(data, columns=["episode_index", "observation.state", "action"])
    sel = [i for i, e in enumerate(t.column("episode_index").to_pylist()) if e == a.episode]
    state = [t.column("observation.state")[i].as_py() for i in sel]
    action = [t.column("action")[i].as_py() for i in sel]
    if len(state) != row["length"]:
        raise SystemExit(f"episode has {len(state)} rows, index says {row['length']}")

    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps(state))
    (out / "trajectory.json").write_text(json.dumps(action))
    ticks = list(range(0, max(0, len(state) - CHUNK_STEPS), a.every))[:a.limit]
    videos = []
    for cam in CAMERA_NAMES:
        key = f"observation.images.{cam}"
        src = ds / info["video_path"].format(
            video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
            file_index=row[f"videos/{key}/file_index"])
        t0 = float(row[f"videos/{key}/from_timestamp"])
        t1 = float(row[f"videos/{key}/to_timestamp"])
        for tick in ticks:
            dst = out / "frames" / f"{cam}_{tick:06d}.jpg"
            subprocess.run([a.ffmpeg, "-loglevel", "error", "-y", "-ss",
                            f"{t0 + tick / fps:.6f}", "-i", str(src), "-frames:v", "1",
                            "-q:v", "2", str(dst)], check=True)
        videos.append({"camera": cam, "path": str(src), "sha256": _sha256(src),
                       "fps": fps, "n_frames": int(round((t1 - t0) * fps)),
                       "first_frame_epoch": 0.0,
                       "segment_s": [t0, t1]})
    manifest = {
        "schema": MANIFEST_SCHEMA_ID,
        "episode_id": f"{ds.name}:ep{a.episode:03d}",
        "task": {"instruction": (row.get("tasks") or [""])[0]},
        "checkpoint": {"id": None, "note": "recorded human demonstration"},
        "timestamps": {"control_hz": fps, "recorded_start_epoch": 0.0,
                       "basis": "LeRobot episode-relative timestamps"},
        "videos": videos,
        "state": {"path": "state.json", "sha256": _sha256(out / "state.json"),
                  "n_rows": len(state), "first_row_epoch": 0.0,
                  "names": list(ACTION_NAMES)},
        "trajectory": {"path": "trajectory.json",
                       "sha256": _sha256(out / "trajectory.json"),
                       "provenance": "recorded_demo"},
        "calibration": {"note": "dataset rig, not this rig"},
        "outcome": {"success": None, "source": f"{ds.name} (not verified here)"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"episode {a.episode}: {len(state)} ticks, task {manifest['task']['instruction']!r}")
    print(f"frames for {len(ticks)} tick(s) x {len(CAMERA_NAMES)} cameras -> {out}/frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
