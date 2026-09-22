"""Export one recorded LeRobot v3 episode into the kuka episode_manifest.v1 layout.

    python3 make_episode.py <dataset_dir> <episode_index> <out_dir> [--every 50]

Writes <out_dir>/manifest.json, state.json, trajectory.json, media/{base,wrist}.mp4,
media/{state,trajectory}.parquet and frames/<cam>_<idx:06d>.png at every sampled tick.
The episode start epoch comes from the source mcap filename (1 s resolution, local
time of the recording box). Video frame i and state row i are the same LeRobot
frame, so the two streams share one epoch by construction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

CAMS = ("base", "wrist")
NAMES = ["A1", "A2", "A3", "A4", "A5", "A6", "gripper"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nb_frames(p: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                   "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)])
    return int(out.decode().strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("episode", type=int)
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--chunk-steps", type=int, default=50)
    ap.add_argument("--task-name", default="dishwasher_door_open")
    ap.add_argument("--tz", default="Asia/Shanghai")
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    info = json.loads((ds / "meta/info.json").read_text())
    fps = float(info["fps"])

    eps = pd.concat([pq.read_table(f).to_pandas()
                     for f in sorted(glob.glob(str(ds / "meta/episodes/chunk-*/*.parquet")))])
    ep = eps[eps.episode_index == a.episode].iloc[0]
    n = int(ep["length"])
    mcap = (ds / "_converted_mcaps.txt").read_text().split()[a.episode]
    stamp = "_".join(Path(mcap).stem.split("_")[-2:])                      # 20260815_054805
    start = dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=ZoneInfo(a.tz)).timestamp()

    data_file = ds / info["data_path"].format(chunk_index=int(ep["data/chunk_index"]),
                                              file_index=int(ep["data/file_index"]))
    df = pq.read_table(data_file).to_pandas()
    df = df[df.episode_index == a.episode].sort_values("frame_index").reset_index(drop=True)
    assert len(df) == n, (len(df), n)
    assert list(df.frame_index) == list(range(n)), "frame_index not contiguous"

    (out / "media").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    state = [[float(x) for x in r] for r in df["observation.state"]]
    traj = [[float(x) for x in r] for r in df["action"]]
    st = pd.DataFrame({"timestamp": df.timestamp, **{n_: [r[i] for r in state] for i, n_ in enumerate(NAMES)}})
    tr = pd.DataFrame({"timestamp": df.timestamp, **{n_: [r[i] for r in traj] for i, n_ in enumerate(NAMES)}})
    st.to_parquet(out / "media/state.parquet", index=False)
    tr.to_parquet(out / "media/trajectory.parquet", index=False)
    (out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (out / "trajectory.json").write_text(json.dumps(traj), encoding="utf-8")

    ticks = list(range(0, max(0, n - a.chunk_steps), a.every))
    videos = []
    for cam in CAMS:
        key = f"observation.images.{cam}"
        src = ds / info["video_path"].format(video_key=key,
                                             chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                                             file_index=int(ep[f"videos/{key}/file_index"]))
        t0, t1 = float(ep[f"videos/{key}/from_timestamp"]), float(ep[f"videos/{key}/to_timestamp"])
        dst = out / f"media/{cam}.mp4"
        if t0 == 0.0 and nb_frames(src) == n:
            shutil.copyfile(src, dst)                      # one file per episode: copy verbatim
        else:
            subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0}", "-to", f"{t1}",
                                   "-i", str(src), "-c:v", "libx264", "-crf", "18", str(dst)])
        nf = nb_frames(dst)
        sel = "+".join(f"eq(n\\,{t})" for t in ticks)
        tmp = out / "frames" / f"_{cam}_%03d.png"
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-vf", f"select='{sel}'",
                               "-vsync", "0", str(tmp)])
        for j, t in enumerate(ticks, start=1):
            (out / "frames" / f"_{cam}_{j:03d}.png").rename(out / "frames" / f"{cam}_{t:06d}.png")
        w, h = info["features"][key]["shape"][1], info["features"][key]["shape"][0]
        videos.append({"camera": cam, "path": f"media/{cam}.mp4", "sha256": sha256(dst),
                       "n_frames": nf, "fps": fps, "width": w, "height": h,
                       "first_frame_epoch": start})

    cfg_ckpt = {"id": "pi05_corrected_b8/132000",
                "sha256": "c881c42551bfc173064f2c9f33a1ff7c61a7dfc1ad49172dc1d6c384111d72d0",
                "action_space": "absolute_joint_targets_deg", "action_names": NAMES}
    manifest = {
        "schema": "hybrid_rollout.robodojo.kuka.episode_manifest.v1",
        "episode_id": f"{ds.name.split('_table_')[-1]}_ep{a.episode:04d}",
        "dataset": ds.name,
        "task": {"instruction": str(ep["tasks"][0]), "name": a.task_name},
        "checkpoint": cfg_ckpt,
        "timestamps": {"recorded_start_epoch": start, "recorded_end_epoch": start + n / fps,
                       "control_hz": fps,
                       "timezone": f"{a.tz} (from mcap filename {mcap}, 1 s resolution)"},
        "videos": videos,
        "state": {"path": "media/state.parquet", "sha256": sha256(out / "media/state.parquet"),
                  "n_rows": n, "columns": ["timestamp"] + NAMES, "first_row_epoch": start},
        "trajectory": {"path": "media/trajectory.parquet",
                       "sha256": sha256(out / "media/trajectory.parquet"),
                       "n_rows": n, "provenance": "recorded_demo"},
        "calibration": {"intrinsics": None, "extrinsics": None, "verified": False},
        "outcome": {"success": None, "source": "unknown", "predicate_value": None,
                    "notes": ("not in the review station's excluded_episodes.json; that is a human "
                              "quality pass, not a measured success predicate")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "episode_id": manifest["episode_id"], "n": n, "ticks": ticks,
                      "instruction": manifest["task"]["instruction"], "mcap": mcap,
                      "videos": [(v["camera"], v["n_frames"]) for v in videos]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
