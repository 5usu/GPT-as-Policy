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
