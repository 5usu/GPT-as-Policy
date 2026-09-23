"""Three-phase experiment: episode sampling, packet building, phase gating.

Synthetic fixtures only. No robot, API, simulator or model call.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ..robodojo_server.gate_assessment import GATE_INSTRUCTION
from .cli import main as cli_main
from .episode import RecordedEpisode, Sample
from .experiment import ManifestError, load_config
from .packet import PROVENANCE_NOTE, build_packet
from .safety import Mode
from .transports import BoundedAstraProposalSource, RecordedTrajectorySource

FIX = pathlib.Path(__file__).parent / "fixtures"


def rows(name):
    return json.loads((FIX / name).read_text())


def episode(manifest="episode_aligned.json", **kw):
    return RecordedEpisode.open(
        FIX / manifest, state_rows=rows("episode_state.json"),
        traj_rows=rows("episode_trajectory.json"), **kw)


class TestEpisodeSampling:
    def test_opens_a_valid_episode(self):
        ep = episode()
        assert ep.n_ticks == 181 and ep.control_hz == 30.0

    def test_refuses_a_misaligned_episode(self):
        """The whole point: never sample frames that don't match the state."""
        with pytest.raises(ManifestError) as e:
            episode("episode_skewed.json")
        assert "same instants" in str(e.value)

    def test_refuses_a_duration_mismatch(self):
        with pytest.raises(ManifestError):
            episode("episode_length_mismatch.json")

    def test_sample_pairs_tick_with_frames_and_chunk(self):
        s = episode().sample(80, chunk_steps=50)
        assert s.tick == 80
        assert set(s.frames) == {"base", "wrist"}
        assert s.frames["base"]["frame_index"] == 80
        assert len(s.recorded_chunk) == 50
        assert s.chunk_provenance == "recorded_demo"

    def test_chunk_is_the_demo_not_a_model(self):
        s = episode().sample(0)
        assert s.chunk_provenance == "recorded_demo"

    def test_sampling_stops_before_the_end(self):
        samples = episode().sample_ticks(every=40, chunk_steps=50)
        assert all(s.tick + 50 <= 181 for s in samples)

    def test_out_of_range_tick_refused(self):
        with pytest.raises(ManifestError):
            episode().sample(9999)

    def test_state_rows_required(self):
        ep = RecordedEpisode.open(FIX / "episode_aligned.json")
        with pytest.raises(ManifestError) as e:
            ep.sample(0)
        assert "state rows not loaded" in str(e.value)


class TestPacketUsesUpstreamGate:
    def pkt(self, provenance="recorded_demo"):
        s = episode().sample(80)
        return build_packet(task_instruction="open the white dishwasher on the table",
                            observation_id=s.observation_id, state=s.state,
                            chunk=s.recorded_chunk, provenance=provenance,
                            frames=s.frames)

    def test_gate_instruction_is_upstream_verbatim(self):
        """Matches skill/gate_prompt.md's declared sha256 in SOURCE.json."""
        assert hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest() == \
            json.loads((pathlib.Path(__file__).parents[1] / "SOURCE.json").read_text())["gate_sha256"]

    def test_packet_embeds_the_unmodified_gate(self):
        assert GATE_INSTRUCTION in self.pkt()["system"]

    def test_packet_sends_nothing(self):
        assert self.pkt()["sends_nothing"] is True

    def test_provenance_is_explicit_and_distinct(self):
        demo = self.pkt("recorded_demo")["user_text"]
        model = self.pkt("model_predicted")["user_text"]
        assert "RECORDED HUMAN DEMONSTRATION" in demo
        assert "has NOT been executed" in model
        assert demo != model

    def test_unknown_provenance_refused(self):
        with pytest.raises(ValueError):
            self.pkt("something_else")

    def test_schema_offers_upstream_modes(self):
        assert self.pkt()["response_schema"]["properties"]["mode"]["enum"] == \
            ["student", "edit", "eef", "stop"]

    def test_eef_is_warned_as_refused(self):
        assert "REFUSED by the execution gate" in self.pkt()["system"]

    def test_no_image_bytes_embedded(self):
        p = self.pkt()
        blob = json.dumps(p)
        assert "base64" not in blob and "data:image" not in blob

    def test_normalisation_artefact_explained(self):
        assert "quantile" in self.pkt()["system"]


class TestPhaseSources:
    def test_phase2_uses_the_demo_not_the_model(self):
        s = RecordedTrajectorySource({"o": [[1.0] * 7]})
        assert s.provenance == "recorded_demo"
        assert s.propose({"observation_id": "o"})["provenance"] == "recorded_demo"

    def test_phase2_missing_observation_is_an_error(self):
        assert RecordedTrajectorySource({}).propose({"observation_id": "x"})["ok"] is False

    def test_phase3_refuses_without_bounds(self):
        r = BoundedAstraProposalSource(proposer=lambda o, b: [[0.0] * 7]).propose({})
        assert r["ok"] is False and "bounded_action_space" in r["error"]

    def test_phase3_refuses_without_proposer(self):
        r = BoundedAstraProposalSource(bounded_action_space={"a": 1}).propose({})
        assert r["ok"] is False


class TestCliGating:
    ARGS = ["--manifest", str(FIX / "episode_aligned.json"),
            "--state-json", str(FIX / "episode_state.json"),
            "--trajectory-json", str(FIX / "episode_trajectory.json")]

    def test_preflight_reports_missing(self, capsys):
        assert cli_main(["preflight"]) == 0
        out = capsys.readouterr().out
        assert "16 missing" in out and "20 missing" in out

    def test_phase1_writes_packets_and_sends_nothing(self, tmp_path, capsys):
        out = tmp_path / "p.jsonl"
        assert cli_main(["phase1", *self.ARGS, "--every", "60", "--limit", "2",
                         "--out", str(out)]) == 0
        assert len(out.read_text().strip().splitlines()) == 2
        assert "NOTHING SENT" in capsys.readouterr().out

    def test_send_flag_refuses(self, tmp_path):
        assert cli_main(["phase1", *self.ARGS, "--send",
                         "--out", str(tmp_path / "p.jsonl")]) == 2

    def test_phase2_arm_refuses_with_missing_config(self, tmp_path):
        assert cli_main(["phase2", *self.ARGS, "--arm",
                         "--audit", str(tmp_path / "a.jsonl")]) == 2

    def test_phase2_shadow_sends_nothing(self, tmp_path, capsys):
        assert cli_main(["phase2", *self.ARGS, "--every", "60", "--limit", "2",
                         "--audit", str(tmp_path / "a.jsonl")]) == 0
        assert "commands sent: 0" in capsys.readouterr().out

    def test_phase3_refuses(self, capsys):
        assert cli_main(["phase3"]) == 2
        assert "REFUSED" in capsys.readouterr().out


class TestLiveSourcesAreOffByDefault:
    """PROPERTY: nothing live happens unless explicitly switched on."""

    def test_astra_defaults_to_dry_run(self):
        from .transports import AstraReviewSource
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="NOPE")
        assert a.enabled is False and a.dry_run is True
        assert a.preflight()[0] is False

    def test_dry_run_returns_a_hash_and_sends_nothing(self):
        from .transports import AstraReviewSource
        called = []
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="NOPE",
                              transport=lambda *args: called.append(args))
        pkt = build_packet(task_instruction="t", observation_id="o",
                           state=[0.0] * 7, chunk=[[0.0] * 7],
                           provenance="model_predicted", frames={})
        r = a.review(pkt)
        assert r["dry_run"] is True and len(r["body_sha256_12"]) == 12
        assert called == [], "dry run must not reach the transport"

    def test_live_without_key_refuses(self, monkeypatch):
        from .transports import AstraReviewSource
        monkeypatch.delenv("NO_SUCH_KEY", raising=False)
        a = AstraReviewSource(base_url="https://x", model="m",
                              api_key_env="NO_SUCH_KEY", enabled=True, dry_run=False)
        pkt = build_packet(task_instruction="t", observation_id="o",
                           state=[0.0] * 7, chunk=[[0.0] * 7],
                           provenance="model_predicted", frames={})
        assert a.review(pkt)["ok"] is False

    def test_body_never_contains_the_credential(self, monkeypatch):
        from .transports import AstraReviewSource
        monkeypatch.setenv("FAKE_KEY", "super-secret-value")
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="FAKE_KEY")
        pkt = build_packet(task_instruction="t", observation_id="o",
                           state=[0.0] * 7, chunk=[[0.0] * 7],
                           provenance="model_predicted", frames={})
        assert "super-secret-value" not in json.dumps(a.build_body(pkt))

    def test_local_source_refuses_a_wrong_checkpoint(self):
        """use_relative_actions=False means a different training run."""
        from .transports import LocalPi05ProposalSource
        src = LocalPi05ProposalSource(chunks={"o": [[0.0] * 7] * 50},
                                      meta={"use_relative_actions": False})
        r = src.propose({"observation_id": "o"})
        assert r["ok"] is False and "different training run" in r["error"]

    def test_local_source_accepts_the_right_checkpoint(self):
        from .transports import LocalPi05ProposalSource
        src = LocalPi05ProposalSource(chunks={"o": [[1.0] * 7] * 50},
                                      checkpoint_id="ck",
                                      meta={"use_relative_actions": True})
        r = src.propose({"observation_id": "o"})
        assert r["ok"] is True and r["is_live"] is True
        assert r["provenance"] == "model_predicted"

    def test_local_source_refuses_a_malformed_chunk(self):
        from .transports import LocalPi05ProposalSource
        src = LocalPi05ProposalSource(chunks={"o": [[0.0] * 7] * 10},
                                      meta={"use_relative_actions": True})
        assert "expected 50 steps" in src.propose({"observation_id": "o"})["error"]

    def test_local_source_needs_a_source(self):
        from .transports import LocalPi05ProposalSource
        with pytest.raises(ValueError):
            LocalPi05ProposalSource()

    def test_the_package_still_refuses_to_load_a_policy(self):
        """Narrowed deliberately, and the reason matters.

        The old rule was that no pi05_serve module may exist at all. That rule
        left jetson/pi05_serve_jetson.py importing a module that was not
        committed, so the one launcher able to serve this checkpoint has been
        failing with ImportError -- which is why "live pi0.5" has meant
        replaying precomputed chunks.

        The principle worth keeping is narrower than the old rule: this package
        must not decide HOW the checkpoint is loaded or run. pi05_serve supplies
        the HTTP surface the committed launcher expects and nothing else; its
        load_policy refuses, and the deployment engineer's loader is injected
        over the top.
        """
        from . import pi05_serve
        with pytest.raises(NotImplementedError) as e:
            pi05_serve.load_policy("/some/checkpoint")
        assert "no loader installed" in str(e.value)
        assert pi05_serve.predict({"state": []})["ok"] is False

    def test_the_serving_layer_enforces_the_checkpoint_contract(self):
        """The eight look-alike finetunes load cleanly and return the wrong
        space. That must be refused at load, not discovered on the arm."""
        from .pi05_serve import ContractViolation, check_contract
        check_contract({"use_relative_actions": True})
        with pytest.raises(ContractViolation) as e:
            check_contract({"use_relative_actions": False,
                            "checkpoint": "/models/looks_right"})
        assert "WRONG SPACE" in str(e.value)

    def test_the_serving_layer_refuses_a_malformed_chunk(self):
        from .pi05_serve import validate_rows
        good = {"ok": True, "rows": [[0.0] * 7 for _ in range(50)]}
        assert validate_rows(good) is good
        assert "expected 50 steps" in validate_rows(
            {"ok": True, "rows": [[0.0] * 7]})["error"]
        assert "values, expected 7" in validate_rows(
            {"ok": True, "rows": [[0.0] * 6 for _ in range(50)]})["error"]
        bad = [[0.0] * 7 for _ in range(50)]
        bad[3][2] = float("inf")
        assert "row 3" in validate_rows({"ok": True, "rows": bad})["error"]

class TestStreamingSurvivesALongReview:
    """PROPERTY: a long review still completes.

    SUPERSEDED IN THE FIELD -- kept because the SSE parsing it covers is still
    used, but streaming is no longer the answer to the proxy. Later measurement
    on the Jetson put the cutoff at ~62 s, not ~108 s, and found a streamed
    review hanging INDEFINITELY rather than returning: urllib's timeout is per
    socket read, so keep-alives reset it forever. Background submit/poll
    replaces this; see TestAstraBackgroundMode in test_kuka_gates.py."""

    def sse(self, response):
        import json as _json
        return [b"event: response.output_text.delta\n",
                b'data: {"type": "response.output_text.delta", "delta": "{"}\n',
                b"\n",
                b"event: response.completed\n",
                ('data: ' + _json.dumps({"type": "response.completed",
                                         "response": response})).encode() + b"\n",
                b"data: [DONE]\n"]

    def test_final_response_is_taken_from_the_completed_event(self):
        from .transports import _final_from_sse
        want = {"output_text": '{"mode": "student"}', "usage": {"input_tokens": 7}}
        assert _final_from_sse(self.sse(want)) == want

    def test_deltas_and_junk_lines_are_ignored(self):
        from .transports import _final_from_sse
        lines = [b": keep-alive\n", b"\n", b"data: not json\n"] + self.sse({"output_text": "x"})
        assert _final_from_sse(lines) == {"output_text": "x"}

    def test_an_unfinished_stream_yields_no_response(self):
        from .transports import _final_from_sse
        assert _final_from_sse([b'data: {"type": "response.output_text.delta"}\n']) == {}

    def test_streaming_is_declared_in_the_hashed_body(self, monkeypatch):
        from .transports import AstraReviewSource
        monkeypatch.setenv("FAKE_KEY", "not-a-real-key")
        pkt = build_packet(task_instruction="t", observation_id="o", state=[0.0] * 7,
                           chunk=[[0.0] * 7], provenance="model_predicted", frames={})
        sent = []
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="FAKE_KEY",
                              enabled=True, dry_run=False, stream=True,
                              transport=lambda url, body, headers, timeout: sent.append(body) or {})
        a.review(pkt)
        assert sent and sent[0]["stream"] is True, "what is sent must be what is hashed"


class TestBoundedTransportRetry:
    """PROPERTY: retries are for a connection that died, never for an answer we
    dislike. A decision that arrived is final, however it reads."""

    def pkt(self):
        return build_packet(task_instruction="t", observation_id="o", state=[0.0] * 7,
                            chunk=[[0.0] * 7], provenance="model_predicted", frames={})

    def source(self, transport, attempts, monkeypatch):
        from .transports import AstraReviewSource
        monkeypatch.setenv("FAKE_KEY", "not-a-real-key")
        return AstraReviewSource(base_url="https://x", model="m", api_key_env="FAKE_KEY",
                                 enabled=True, dry_run=False, attempts=attempts,
                                 transport=transport)

    def test_default_is_still_a_single_attempt(self, monkeypatch):
        calls = []

        def boom(*a):
            calls.append(1)
            raise RuntimeError("connection reset")
        r = self.source(boom, 1, monkeypatch).review(self.pkt())
        assert len(calls) == 1 and r["ok"] is False

    def test_a_dead_connection_is_retried_up_to_the_bound(self, monkeypatch):
        calls = []

        def flaky(*a):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("stream ended without a completed response event")
            return {"output_text": '{"mode": "stop", "reason": "unsafe"}',
                    "usage": {"total_tokens": 5}}
        r = self.source(flaky, 3, monkeypatch).review(self.pkt())
        assert len(calls) == 3
        assert r["ok"] is True and r["decision"]["mode"] == "stop"
        assert r["attempts_used"] == 3

    def test_an_answered_review_is_never_retried(self, monkeypatch):
        calls = []

        def stopper(*a):
            calls.append(1)
            return {"output_text": '{"mode": "stop", "reason": "the scene is unsafe"}'}
        r = self.source(stopper, 3, monkeypatch).review(self.pkt())
        assert len(calls) == 1, "a decision must never be re-asked"
        assert r["decision"]["mode"] == "stop"

    def test_a_non_json_answer_is_not_retried_either(self, monkeypatch):
        calls = []

        def prose(*a):
            calls.append(1)
            return {"output_text": "I think you should stop"}
        r = self.source(prose, 3, monkeypatch).review(self.pkt())
        assert len(calls) == 1 and r["ok"] is False
