"""The pi0.5 contract end to end: server on loopback, fake policy, real client.

No openpi, no weights, no GPU. The server's HTTP surface, its checkpoint
identification and the client's refusal logic are what is under test.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading

import pytest

from .conftest import MEAN_STATE, chunk, good_meta
from .contract import CAMERA_NAMES, CHECKPOINT
from .pi05_serve import (NORM_STATS_REL, ContractViolation, Server, checkpoint_meta,
                         decode_image, make_handler)
from .transports import (Pi05ChunkReplay, Pi05HttpProposalSource,
                         YamAstraReviewSource, check_checkpoint_meta, rows_ok)


def jpeg_b64(w=64, h=36):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    """A checkpoint dir whose norm stats digest the contract is patched to accept."""
    stats = tmp_path / "snapshots" / CHECKPOINT["revision"] / NORM_STATS_REL
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {"note": "test"}}))
    monkeypatch.setitem(CHECKPOINT, "norm_stats_sha256",
                        hashlib.sha256(stats.read_bytes()).hexdigest())
    return tmp_path / "snapshots" / CHECKPOINT["revision"]


class FakePolicy:
    def __init__(self, rows):
        self.rows, self.seen = rows, []

    def infer(self, batch):
        self.seen.append(batch)
        return {"actions": self.rows}


@pytest.fixture
def served(ckpt):
    from http.server import ThreadingHTTPServer
    pol = FakePolicy(chunk())
    srv = Server(pol, checkpoint_meta(ckpt))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(srv))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/infer", pol, srv
    httpd.shutdown()
    httpd.server_close()


def obs(**kw):
    o = {"observation_id": "o1", "state": MEAN_STATE, "task": "stack the blocks",
         "images": {c: jpeg_b64() for c in CAMERA_NAMES}}
    o.update(kw)
    return o


class TestCheckpointIdentity:
    def test_meta_reads_revision_from_hf_snapshot_path(self, ckpt):
        m = checkpoint_meta(ckpt)
        assert m["revision"] == CHECKPOINT["revision"] and m["revision_matches"]
        assert check_checkpoint_meta(m) is None

    def test_wrong_norm_stats_refused(self, tmp_path):
        p = tmp_path / NORM_STATS_REL
        p.parent.mkdir(parents=True)
        p.write_text("{}")
        with pytest.raises(ContractViolation):
            checkpoint_meta(tmp_path)

    def test_not_a_checkpoint_refused(self, tmp_path):
        with pytest.raises(ContractViolation):
            checkpoint_meta(tmp_path)

    @pytest.mark.parametrize("key,val", [("repo_id", "someone/pi0.5-yam-ft"),
                                         ("action_horizon", 50), ("action_dim", 7),
                                         ("gripper_convention", "closed_1_open_0"),
                                         ("norm_stats_sha256", "0" * 64)])
    def test_client_refuses_a_different_checkpoint(self, key, val):
        assert check_checkpoint_meta(good_meta(**{key: val}))

    def test_client_refuses_missing_meta(self):
        assert "no checkpoint meta" in check_checkpoint_meta(None)

    def test_rows_ok(self):
        assert rows_ok(chunk()) is None
        assert rows_ok(chunk()[:10]) and rows_ok([[0.0] * 7] * 16)
        bad = chunk(); bad[0][0] = float("nan")
        assert rows_ok(bad)


class TestServerEndToEnd:
    def test_round_trip(self, served):
        url, pol, _ = served
        out = Pi05HttpProposalSource(url).propose(obs())
        assert out["ok"], out
        assert len(out["rows"]) == 16 and len(out["rows"][0]) == 14
        assert out["checkpoint_id"].startswith("robocurve/pi0.5-yam@ee17bb36")
        b = pol.seen[0]
        assert list(b["images"]) == list(CAMERA_NAMES)
        assert b["images"]["top"].shape == (36, 64, 3) and b["prompt"] == "stack the blocks"
        assert b["state"].shape == (14,)

    def test_missing_camera_refused_before_inference(self, served):
        url, pol, _ = served
        o = obs(); del o["images"]["right"]
        out = Pi05HttpProposalSource(url).propose(o)
        assert not out["ok"] and "right" in out["error"] and not pol.seen

    def test_empty_prompt_refused(self, served):
        url, pol, _ = served
        assert not Pi05HttpProposalSource(url).propose(obs(task=""))["ok"]

    def test_short_state_refused(self, served):
        url, _, _ = served
        assert not Pi05HttpProposalSource(url).propose(obs(state=MEAN_STATE[:7]))["ok"]

    def test_client_refuses_a_server_serving_something_else(self, served):
        url, _, srv = served
        srv.meta = {**srv.meta, "repo_id": "other/yam"}
        out = Pi05HttpProposalSource(url).propose(obs())
        assert not out["ok"] and "does not match" in out["error"]

    def test_health(self, served):
        url, _, _ = served
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        h = json.loads(opener.open(url.replace("/infer", "/health")).read())
        assert h["ok"] and h["meta"]["repo_id"] == CHECKPOINT["repo_id"]

    def test_unreachable_is_an_error_not_a_chunk(self):
        out = Pi05HttpProposalSource("http://127.0.0.1:9/infer", timeout_s=0.5).propose(obs())
        assert not out["ok"]


class TestReplay:
    def test_keyed(self):
        src = Pi05ChunkReplay({"o1": chunk()}, good_meta())
        assert src.propose({"observation_id": "o1"})["provenance"] == "model_predicted"
        assert not src.propose({"observation_id": "o2"})["ok"]

    def test_sequential_is_labelled_replay(self):
        src = Pi05ChunkReplay({"a": chunk(), "b": chunk()}, good_meta(), sequential=True)
        out = src.propose({"observation_id": "live:9"})
        assert out["provenance"] == "recorded_chunk_replay" and out["replayed_from"] == "a"
        assert src.propose({})["replayed_from"] == "b"

    def test_wrong_meta_refused(self):
        src = Pi05ChunkReplay({"o1": chunk()}, good_meta(action_dim=7))
        assert not src.propose({"observation_id": "o1"})["ok"]

    def test_from_file(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"_meta": good_meta(), "o1": chunk()}))
        assert Pi05ChunkReplay.from_file(str(p)).propose({"observation_id": "o1"})["ok"]


def test_decode_image_accepts_data_url_and_arrays():
    a = decode_image("data:image/jpeg;base64," + jpeg_b64())
    assert a.shape == (36, 64, 3)
    assert decode_image([[[0, 0, 0]]]).shape == (1, 1, 3)


class TestAstraClient:
    def test_dry_run_sends_nothing_and_names_the_schema(self):
        from .packet import build_packet
        rv = YamAstraReviewSource(base_url="https://invalid.example/v1/responses",
                                  model="gpt-6-astra", api_key_env="NOPE", dry_run=True)
        pkt = build_packet(task_instruction="t", observation_id="o", state=MEAN_STATE,
                           chunk=chunk(), provenance="model_predicted", frames={})
        body = rv.build_body(pkt)
        assert body["text"]["format"]["name"] == "yam_action_review"
        out = rv.review(pkt)
        assert out["dry_run"] and not out["ok"]
