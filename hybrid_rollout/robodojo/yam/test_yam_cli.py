"""CLI refusals and the recorded-episode loop, offline."""
from __future__ import annotations

import json

from .cli import main
from .conftest import chunk, good_meta
from .episode import YamRecordedEpisode


def test_preflight_runs(capsys):
    assert main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "robocurve/pi0.5-yam" in out and "`execute` only" in out


def test_hold_and_live_refuse_without_hold(capsys):
    assert main(["hold"]) == 2
    assert main(["live", "--pi05-url", "http://127.0.0.1:1/infer"]) == 2
    assert "ENABLES THE MOTORS" in capsys.readouterr().err


def test_run_refuses_without_a_proposal_source(episode_dir):
    assert main(["run", "--episode", str(episode_dir)]) == 2


def test_phase3_refuses():
    assert main(["phase3"]) == 2


def test_episode_samples_sixteen_step_chunks(episode_dir):
    ep = YamRecordedEpisode.open(
        episode_dir / "manifest.json", media_root=episode_dir,
        state_rows=json.loads((episode_dir / "state.json").read_text()),
        traj_rows=json.loads((episode_dir / "trajectory.json").read_text()))
    s = ep.sample_ticks(every=16)
    assert len(s) == 3 and len(s[0].recorded_chunk) == 16
    assert all(s[0].frames[c]["frame_present"] for c in ("top", "left", "right"))


def test_phase1_writes_packets(episode_dir, tmp_path, capsys):
    out = tmp_path / "p.jsonl"
    assert main(["phase1", "--episode", str(episode_dir), "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["provenance"] == "recorded_demo"
    assert "NOTHING SENT" in capsys.readouterr().out


def test_run_with_chunks_and_mock_free_dry_run(episode_dir, tmp_path, capsys):
    ids = [f"yam_test_ep:t{t:06d}" for t in (0, 16, 32)]
    cf = tmp_path / "chunks.json"
    cf.write_text(json.dumps({"_meta": good_meta(), **{i: chunk() for i in ids}}))
    audit = tmp_path / "audit.jsonl"
    rc = main(["run", "--episode", str(episode_dir), "--chunks-file", str(cf),
               "--audit", str(audit)])
    assert rc == 0
    rows = [json.loads(l) for l in audit.read_text().splitlines()]
    assert len(rows) == 3
    # Astra is a dry run: the request is built, nothing is sent, no decision
    assert all(r["outcome"] == "no_decision" and r["review"]["dry_run"] for r in rows)
    assert all(len(r["proposal"]["values"][0]) == 14 for r in rows)
    assert "commands sent: 0" in capsys.readouterr().out


def test_run_refuses_a_foreign_chunks_file(episode_dir, tmp_path):
    cf = tmp_path / "chunks.json"
    cf.write_text(json.dumps({"_meta": good_meta(repo_id="x/y"),
                              "yam_test_ep:t000000": chunk()}))
    audit = tmp_path / "audit.jsonl"
    main(["run", "--episode", str(episode_dir), "--chunks-file", str(cf),
          "--audit", str(audit), "--limit", "1"])
    row = json.loads(audit.read_text().splitlines()[0])
    assert row["outcome"] == "no_proposal" and "does not match" in row["reason"]
