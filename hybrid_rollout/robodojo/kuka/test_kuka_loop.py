"""Loop, eef gating, success sourcing, alignment, retries and idempotency.

Fake transports only. No real robot, API, simulator or model call.
"""
from __future__ import annotations

import json
import pathlib
import time

import pytest

from .contract import EEF_PREREQUISITES, eef_execution_gate
from .experiment import (CameraAssessment, assess_success, frame_for_tick,
                         load_manifest, validate_manifest)
from .loop import AuditLog, KukaReviewLoop, Outcome, Stage, evaluate_stop_signals
from .safety import ArmingRefused, CommandLedger, Mode, Supervisor
from .schema import response_schema
from .test_kuka_gates import (FIX, ROBOT, SECRET, STATE, armable_raw_config,
                              chunks, decisions, full_config, loop, observation,
                              raw_config)
from .transports import FakeKukaGateway, ShadowGateway, a800_live_review_enabled
from .validation import execution_eligible, validate_chunk


class TestEefAcceptedButRefused:
    """PROPERTY: eef stays in the upstream contract and never becomes a command."""

    def test_eef_is_in_the_public_schema(self):
        s = response_schema()
        assert "eef" in s["properties"]["mode"]["enum"]
        assert "target" in s["properties"]

    def test_eef_parses_as_a_decision(self):
        d = decisions()["obs-eef"]
        assert d["mode"] == "eef" and "target" in d

    def test_eef_cannot_emit_a_kuka_command(self):
        """The headline test: parsed, recorded, refused, nothing sent."""
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw, oid="obs-eef",
                   decisions_override={**decisions(),
                                       "obs-feasible": decisions()["obs-eef"]}
                   ).step(observation("obs-feasible"))
        assert rec.outcome == Outcome.EEF_REFUSED.value
        assert rec.decision["eef_gate"]["accepted_as_upstream_decision"] is True
        assert rec.decision["eef_gate"]["execution_allowed"] is False
        assert rec.envelope is None
        assert gw.received == [], "eef must never reach the gateway"

    def test_all_four_prerequisites_are_listed_as_missing(self):
        allowed, missing, _ = eef_execution_gate()
        assert allowed is False
        assert set(missing) == set(EEF_PREREQUISITES)

    @pytest.mark.parametrize("key", sorted(EEF_PREREQUISITES))
    def test_each_prerequisite_is_individually_required(self, key):
        assert EEF_PREREQUISITES[key] is False


class TestSuccessCannotComeFromProse:
    """PROPERTY: a model's words never constitute task success."""

    def cam(self, conf=0.99):
        return CameraAssessment(progress="the door looks fully open",
                                evidence_frames=[{"camera": "base", "frame": 120}],
                                confidence=conf, prose="clearly open, high confidence")

    def test_confident_prose_yields_unconfirmed(self):
        out = assess_success(camera=self.cam(), predicate_config=None)
        assert out["success"] is None and out["source"] == "unknown"

    def test_predicate_configured_but_unevaluated_is_not_success(self):
        out = assess_success(camera=self.cam(), predicate_config="door_angle>60",
                             predicate_value=None)
        assert out["success"] is None

    def test_measured_predicate_is_accepted(self):
        out = assess_success(camera=self.cam(), predicate_config="door_angle>60",
                             predicate_value=True)
        assert out["success"] is True and out["source"] == "measured_predicate"

    def test_supervisor_confirmation_is_accepted(self):
        out = assess_success(camera=self.cam(), supervisor_confirmed=True)
        assert out["success"] is True and out["source"] == "supervisor_confirmed"

    def test_camera_assessment_requires_evidence_and_confidence(self):
        assert CameraAssessment("x", [], 0.9).valid()[0] is False
        assert CameraAssessment("x", [{"frame": 1}], None).valid()[0] is False
        assert CameraAssessment("x", [{"frame": 1}], 0.5).valid()[0] is True

    def test_assessment_log_declares_itself_insufficient(self):
        assert self.cam().to_log()["sufficient_for_success"] is False

    def test_loop_records_success_as_unconfirmed(self):
        rec = loop(Mode.REPLAY).step(observation(), camera_assessment=self.cam())
        assert rec.success["success"] is None


class TestManifestAlignment:
    """PROPERTY: frames and state rows must describe the same instants."""

    def test_aligned_manifest_passes(self):
        assert validate_manifest(load_manifest(FIX / "episode_aligned.json")) == []

    def test_skewed_timestamps_rejected(self):
        problems = validate_manifest(load_manifest(FIX / "episode_skewed.json"))
        assert any("from the first state row" in p for p in problems)

    def test_duration_mismatch_rejected(self):
        problems = validate_manifest(load_manifest(FIX / "episode_length_mismatch.json"))
        assert any("different durations" in p for p in problems)

    def test_missing_field_rejected(self):
        m = load_manifest(FIX / "episode_aligned.json")
        del m.raw["state"]
        assert any("state" in p for p in validate_manifest(m))

    def test_trajectory_must_be_recorded_demo(self):
        m = load_manifest(FIX / "episode_aligned.json")
        m.raw["trajectory"]["provenance"] = "model_predicted"
        assert any("recorded_demo" in p for p in validate_manifest(m))

    def test_tick_to_frame_mapping(self):
        m = load_manifest(FIX / "episode_aligned.json")
        assert frame_for_tick(m, "base", 0) == 0
        assert frame_for_tick(m, "base", 80) == 80

    def test_manifest_has_no_media_bytes(self):
        """Videos are referenced by path+checksum, never embedded."""
        raw = json.loads((FIX / "episode_aligned.json").read_text())
        for v in raw["videos"]:
            assert set(v) >= {"path", "sha256"}
            assert "data" not in v and "bytes" not in v


class TestStopAndRetry:
    """PROPERTY: bounded retries, and every configured stop condition halts."""

    @pytest.mark.parametrize("flag,cond", [
        ("handle_visible", "lost_handle"),
        ("load_exceeded", "unexpected_contact_or_load"),
        ("door_outside_region", "door_outside_expected_region"),
        ("vision_stale", "stale_vision"),
        ("heartbeat_lost", "heartbeat_lost"),
        ("estop_engaged", "estop_engaged")])
    def test_each_stop_condition_halts(self, flag, cond):
        obs = observation()
        obs[flag] = False if flag == "handle_visible" else True
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=FakeKukaGateway(SECRET)).step(obs)
        assert rec.outcome == Outcome.STOPPED.value
        assert cond in rec.stop_signals

    def test_disagreement_without_tolerance_is_unevaluable_not_silent(self):
        sig = evaluate_stop_signals(
            configured=["commanded_observed_disagreement"], observation={},
            feedback={"ok": True, "commanded": [0] * 6, "measured": [9] * 6},
            tolerance_deg=None)
        assert sig == ["commanded_observed_disagreement:unevaluable"]

    def test_disagreement_detected_when_tolerance_supplied(self):
        sig = evaluate_stop_signals(
            configured=["commanded_observed_disagreement"], observation={},
            feedback={"ok": True, "commanded": [0] * 6, "measured": [9] * 6},
            tolerance_deg=0.5)
        assert sig == ["commanded_observed_disagreement"]

    def test_stop_is_sticky(self):
        lp = loop(Mode.REVIEWED_EXECUTION, gateway=FakeKukaGateway(SECRET))
        obs = observation(); obs["estop_engaged"] = True
        lp.step(obs)
        rec = lp.step(observation())          # clean observation afterwards
        assert rec.outcome == Outcome.STOPPED.value

    def test_retry_limit_stops(self):
        gw = FakeKukaGateway(SECRET)
        lp = loop(Mode.REVIEWED_EXECUTION, gateway=gw,
                  decisions_override={"obs-feasible": decisions()["obs-bigedit"]})
        outcomes = [lp.step(observation()).outcome for _ in range(5)]
        assert Outcome.STOPPED.value in outcomes
        assert outcomes.index(Outcome.STOPPED.value) <= \
            raw_config()["stop_conditions"]["max_retries_per_subgoal"] + 1

    def test_reviewer_stop_halts(self):
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=FakeKukaGateway(SECRET),
                   decisions_override={"obs-feasible": decisions()["obs-stop"]}
                   ).step(observation())
        assert rec.outcome == Outcome.STOPPED.value


class TestSingleIdempotentCommand:
    """PROPERTY: after every gate passes, the arm receives exactly one command."""

    def test_exactly_one_command_one_step(self):
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw).step(observation())
        assert rec.outcome == Outcome.COMMAND_SENT.value
        assert len(gw.received) == 1
        assert gw.received[0]["n_steps"] == 1, "supervised execution is single-step"
        assert gw.received[0]["ipoc"] == 1

    def test_replaying_the_same_envelope_is_refused(self):
        gw = FakeKukaGateway(SECRET)
        lp = loop(Mode.REVIEWED_EXECUTION, gateway=gw)
        rec = lp.step(observation())
        env = lp.ledger.issued()
        assert len(env) == 1
        from .safety import CommandEnvelope
        e = CommandEnvelope(**{k: v for k, v in [
            ("command_id", env[0]), ("mode", Mode.REVIEWED_EXECUTION.value),
            ("robot", ROBOT.key()), ("rows", gw.received[0]["rows"]),
            ("n_steps", 1), ("control_hz", 30.0),
            ("issued_at", rec.envelope["issued_at"]),
            ("observation_id", rec.envelope["observation_id"]),
            ("observation_age_s", rec.envelope["observation_age_s"]),
            ("decision_mode", rec.envelope["decision_mode"]),
            ("audit", rec.envelope["audit"])]})
        e.sign(SECRET); e.approved_for_execution = True
        with pytest.raises(ArmingRefused) as exc:
            gw.send(e)
        assert exc.value.code == "duplicate_envelope"

    def test_ledger_refuses_a_reused_command_id(self):
        ledger = CommandLedger()
        ledger.reserve("cmd-1", "a" * 64)
        with pytest.raises(ArmingRefused) as e:
            ledger.reserve("cmd-1", "b" * 64)
        assert e.value.code == "duplicate_command"

    def test_unsigned_envelope_rejected(self):
        from .safety import CommandEnvelope
        e = CommandEnvelope("c", "reviewed_execution", ROBOT.key(), [[0.0] * 7], 1,
                            30.0, time.time(), "o", 0.0, "student")
        e.approved_for_execution = True
        with pytest.raises(ArmingRefused) as exc:
            FakeKukaGateway(SECRET).send(e)
        assert exc.value.code == "bad_signature"

    def test_unapproved_envelope_rejected(self):
        from .safety import CommandEnvelope
        e = CommandEnvelope("c", "reviewed_execution", ROBOT.key(), [[0.0] * 7], 1,
                            30.0, time.time(), "o", 0.0, "student")
        e.sign(SECRET)
        with pytest.raises(ArmingRefused) as exc:
            FakeKukaGateway(SECRET).send(e)
        assert exc.value.code == "unapproved_envelope"

    def test_stale_observation_refused(self):
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw).step(
            observation(epoch=time.time() - 30.0))
        assert rec.envelope["code"] == "stale_observation"
        assert gw.received == []

    def test_unsafe_candidate_refused(self):
        """The breaching chunk's gripper stays out of range after sanitization?
        No -- sanitization fixes it, so this asserts the gate ran, not that it
        failed. The explicit failure path is covered by execution_eligible."""
        assert execution_eligible(
            validate_chunk([[0.0] * 6 + [1.5]], state=STATE), True)[0] is False

    def test_shipped_config_halts_because_tolerance_is_unset(self):
        """The shipped config cannot finish a cycle: with no measured
        commanded-vs-observed tolerance, disagreement is UNEVALUABLE, and an
        unevaluable safety check must stop rather than pass."""
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw, raw=raw_config()
                   ).step(observation())
        assert rec.outcome == Outcome.STOPPED.value
        assert "commanded_observed_disagreement:unevaluable" in rec.stop_signals

    def test_shadow_gateway_never_sends(self):
        gw = ShadowGateway()
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw).step(observation())
        assert rec.gateway["sent"] is False and gw.emitted[0]["sent"] is False


class TestA800Boundary:
    def test_live_review_off_by_default(self, monkeypatch):
        monkeypatch.delenv("A800_LIVE_REVIEW", raising=False)
        assert a800_live_review_enabled()[0] is False

    def test_flag_without_dedicated_key_still_off(self, monkeypatch):
        monkeypatch.setenv("A800_LIVE_REVIEW", "1")
        monkeypatch.delenv("A800_ASTRA_API_KEY", raising=False)
        ok, why = a800_live_review_enabled()
        assert ok is False and "dedicated key" in why

    def test_does_not_read_the_eng1_key(self, monkeypatch):
        monkeypatch.setenv("A800_LIVE_REVIEW", "1")
        monkeypatch.delenv("A800_ASTRA_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "placeholder-eng1-key-must-not-be-read")
        assert a800_live_review_enabled()[0] is False


class TestAuditFormat:
    def test_all_modes_share_the_row_shape(self, tmp_path):
        keys = None
        for mode in (Mode.REPLAY, Mode.LIVE_SHADOW, Mode.REVIEWED_EXECUTION):
            log = AuditLog(tmp_path / f"{mode.value}.jsonl")
            loop(mode, gateway=FakeKukaGateway(SECRET), audit=log).step(observation())
            row = json.loads(log.path.read_text().splitlines()[0])
            if keys is None:
                keys = set(row)
            assert set(row) == keys, f"{mode.value} row shape differs"

    def test_audit_is_append_only(self, tmp_path):
        p = tmp_path / "a.jsonl"
        for _ in range(3):
            loop(Mode.REPLAY, audit=AuditLog(p)).step(observation())
        assert len(p.read_text().strip().splitlines()) == 3

    def test_envelope_signature_never_logged(self):
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.REVIEWED_EXECUTION, gateway=gw).step(observation())
        assert "signature" not in rec.envelope
        assert rec.envelope["signature_present"] is True


class TestReviewPacketReachesTheReviewer:
    """PROPERTY: the loop hands the reviewer the full gate packet, not just an id."""

    class _Capture:
        name = "capture"
        is_live = False

        def __init__(self):
            self.packets = []

        def review(self, packet):
            self.packets.append(packet)
            return {"ok": True, "decision": dict(decisions()["obs-feasible"])}

    def obs(self):
        return {**observation(), "task": "open the white dishwasher on the table",
                "frames": {"base": {"frame_index": 3, "frame_path": "/m/frames/base_000003.png",
                                    "frame_present": True}},
                "images": {"base": "SU1BR0VCWVRFUw=="},
                "image_data_urls": ["data:image/png;base64,SU1BR0VCWVRFUw=="]}

    def test_packet_carries_gate_prompt_schema_task_and_images(self):
        cap = self._Capture()
        lp = loop(Mode.LIVE_SHADOW)
        lp.review_source = cap
        lp.step(self.obs())
        (p,) = cap.packets
        assert p["request_id"] == "obs-feasible"
        assert {"system", "user_text", "response_schema"} <= set(p)
        assert "open the white dishwasher on the table" in p["user_text"]
        assert "base_000003.png" in p["user_text"]
        assert p["provenance"] == "model_predicted"
        assert p["image_data_urls"] == ["data:image/png;base64,SU1BR0VCWVRFUw=="]

    def test_a_live_astra_source_can_build_a_body_from_the_loop_packet(self):
        from .transports import AstraReviewSource
        sent = []
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="NOPE",
                              transport=lambda *args: sent.append(args))
        lp = loop(Mode.LIVE_SHADOW)
        lp.review_source = a
        rec = lp.step(self.obs())
        assert rec.review["ok"] is False and sent == []     # dry run: built, not sent
        assert rec.outcome == Outcome.NO_DECISION.value

    def test_image_bytes_are_not_written_to_the_audit(self, tmp_path):
        log = AuditLog(tmp_path / "a.jsonl")
        loop(Mode.LIVE_SHADOW, audit=log).step(self.obs())
        text = log.path.read_text()
        assert "SU1BR0VCWVRFUw" not in text
        row = json.loads(text.splitlines()[0])
        assert row["observation"]["images_attached"] == ["base"]

    def test_astra_source_refuses_an_incomplete_packet_instead_of_raising(self):
        from .transports import AstraReviewSource
        a = AstraReviewSource(base_url="https://x", model="m", api_key_env="NOPE")
        r = a.review({"request_id": "o", "mode": "live_shadow"})
        assert r["ok"] is False and "packet" in r["error"]
