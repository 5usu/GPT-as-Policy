"""The four-mode state machine on YAM: stages, locks and arming refusals."""
from __future__ import annotations

import time

import pytest

from ..kuka.transports import FakeKukaGateway
from .conftest import MEAN_STATE, chunk, decision
from .contract import MAX_CORRECTION_RAD
from .experiment import flatten_config, load_config
from .loop import AuditLog, Outcome, Stage, YamReviewLoop, apply_edit, evaluate_stop_signals
from .safety import (ArmingRefused, Mode, RigIdentity, Supervisor, authorise,
                     identity_from_config, missing_config)
from .transports import RecordedReview, ShadowGateway

SECRET = b"s" * 32
RIG = RigIdentity("I2RT YAM", "rig-1", "can_l", "can_r")


class Proposals:
    name, is_live, provenance = "fixed", False, "model_predicted"

    def __init__(self, rows):
        self.rows = rows

    def propose(self, obs):
        return {"ok": True, "rows": self.rows, "source": self.name}


class Review:
    name, is_live = "scripted", False

    def __init__(self, d):
        self.d = d
        self.packets = []

    def review(self, packet):
        self.packets.append(packet)
        return {"ok": True, "decision": self.d}


def measured_config():
    """Every required value, as if measured. Test-only."""
    flat = {k: 1.0 for k in ("max_speed", "max_acceleration", "max_step_displacement",
                             "observation_freshness_s", "heartbeat_timeout_s",
                             "commanded_observed_tolerance_rad")}
    flat.update(camera_mapping={"top": 0, "left": 2, "right": 4},
                gripper_limits_left=[0, 1], gripper_limits_right=[0, 1],
                rest_pose=MEAN_STATE, table_workspace="table", estop_tested="t-1",
                success_predicate="blocks stacked", observation_freshness_s=5.0,
                heartbeat_timeout_s=5.0)
    return flat


def loop(mode, d, *, rows=None, config=None, gateway=None, secret=None,
         supervisor=None, tmp_path=None, raw=None):
    raw = raw or load_config()
    return YamReviewLoop(
        mode=mode, config=config if config is not None else flatten_config(raw),
        raw_config=raw, proposal_source=Proposals(rows or chunk()),
        review_source=Review(d), gateway=gateway or ShadowGateway(),
        audit=AuditLog(tmp_path / "a.jsonl") if tmp_path else None,
        target=RIG, allowlist=[RIG], supervisor=supervisor, secret=secret)


def obs(**kw):
    o = {"observation_id": "ep:t000000", "state": MEAN_STATE, "task": "stack the blocks",
         "epoch": time.time()}
    o.update(kw)
    return o


class TestObservationModes:
    @pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.LIVE_SHADOW])
    def test_stops_at_validate(self, mode, tmp_path):
        lp = loop(mode, decision(), tmp_path=tmp_path)
        rec = lp.step(obs())
        assert rec.outcome == Outcome.OBSERVED_ONLY.value
        assert rec.stage_reached == Stage.VALIDATE.value and rec.envelope is None
        assert rec.robot_model.startswith("I2RT YAM") and rec.schema.endswith("yam.loop.v1")
        assert (tmp_path / "a.jsonl").read_text().count("\n") == 1

    def test_reviewer_saw_the_bimanual_packet(self):
        lp = loop(Mode.LIVE_SHADOW, decision())
        lp.step(obs())
        assert "LEFT and RIGHT" in lp.review_source.packets[0]["system"]

    def test_images_not_in_audit(self):
        rec = loop(Mode.REPLAY, decision()).step(
            obs(images={"top": "x"}, image_data_urls=["data:..."]))
        assert "images" not in rec.observation and rec.observation["images_attached"] == ["top"]


class TestDecisions:
    def test_eef_refused(self):
        rec = loop(Mode.LIVE_SHADOW, decision("eef")).step(obs())
        assert rec.outcome == Outcome.EEF_REFUSED.value

    def test_stop(self):
        lp = loop(Mode.LIVE_SHADOW, decision("stop"))
        assert lp.step(obs()).outcome == Outcome.STOPPED.value and lp.stopped
        assert lp.step(obs()).outcome == Outcome.STOPPED.value

    def test_sound_edit_is_computed_and_locked(self):
        d = decision("edit", steps=3)
        d["edit"]["right"]["delta_joint_rad"] = [0.005, 0, 0, 0, 0, 0]
        d["edit"]["right"]["gripper"] = "closed"
        lp = loop(Mode.REVIEWED_EXECUTION, d)
        rec = lp.step(obs())
        ec = rec.decision["edit_computed"]
        assert ec["accepted"] and not ec["emitted"] and not ec["gripper_applied"]
        assert ec["gripper_requested"]["right"] == "closed"
        assert rec.emitted_mode == "student"

    def test_oversized_edit_rejected(self):
        d = decision("edit")
        d["edit"]["left"]["delta_joint_rad"] = [MAX_CORRECTION_RAD * 2, 0, 0, 0, 0, 0]
        rec = loop(Mode.LIVE_SHADOW, d).step(obs())
        assert rec.improvement_valid is False

    def test_apply_edit_touches_only_that_arm(self):
        rows = chunk()
        out, why = apply_edit(rows, {"left": {"delta_joint_rad": [0.01] * 6}}, 2)
        assert why == "" and len(out) == 2
        assert out[0][0] == pytest.approx(rows[0][0] + 0.01)
        assert out[0][7] == rows[0][7] and out[0][6] == rows[0][6]

    def test_fatal_proposal_is_held(self):
        rows = chunk(); rows[0][1] = -5.0
        rec = loop(Mode.LIVE_SHADOW, decision(), rows=rows).step(obs())
        assert rec.outcome == Outcome.HELD.value


class TestArming:
    def test_shipped_config_refuses(self):
        rec = loop(Mode.REVIEWED_EXECUTION, decision()).step(obs())
        assert rec.outcome == Outcome.ARMING_REFUSED.value
        assert rec.envelope["code"] == "missing_config"
        assert "camera_mapping" in rec.envelope["missing"]

    def test_shipped_config_names_every_yam_value(self):
        miss = missing_config(flatten_config(load_config()), Mode.REVIEWED_EXECUTION)
        assert {"gripper_limits_left", "gripper_limits_right", "rest_pose",
                "commanded_observed_tolerance_rad", "estop_tested"} <= set(miss)

    def test_identity_is_never_defaulted(self):
        with pytest.raises(ArmingRefused):
            identity_from_config(flatten_config(load_config()))

    def test_no_supervisor_refuses(self):
        rec = loop(Mode.REVIEWED_EXECUTION, decision(), config=measured_config(),
                   secret=SECRET).step(obs())
        assert rec.envelope["code"] == "no_heartbeat"

    def armed(self, raw=None):
        gw = FakeKukaGateway(secret=SECRET)
        sup = Supervisor(heartbeat_at=time.time(), deadman_held=True, estop_clear=True)
        lp = loop(Mode.REVIEWED_EXECUTION, decision(steps=5), config=measured_config(),
                  gateway=gw, secret=SECRET, supervisor=sup, raw=raw)
        return lp.step(obs()), gw

    def test_one_signed_step_through_every_gate(self):
        raw = load_config()
        raw["stop_conditions"]["commanded_observed_tolerance_rad"] = 0.05
        rec, gw = self.armed(raw)
        assert rec.outcome == Outcome.COMMAND_SENT.value, rec.reason
        assert len(gw.received) == 1 and len(gw.received[0]["rows"]) == 1
        assert len(gw.received[0]["rows"][0]) == 14

    def test_unset_tolerance_stops_after_the_first_step(self):
        rec, gw = self.armed()
        assert len(gw.received) == 1
        assert rec.outcome == Outcome.STOPPED.value
        assert "commanded_observed_disagreement:unevaluable" in rec.stop_signals

    def test_authorise_refuses_multi_step(self):
        sup = Supervisor(heartbeat_at=time.time(), deadman_held=True, estop_clear=True)
        from .safety import CommandLedger
        with pytest.raises(ArmingRefused) as e:
            authorise(mode=Mode.REVIEWED_EXECUTION, config=measured_config(),
                      target=RIG, allowlist=[RIG], supervisor=sup,
                      ledger=CommandLedger(), command_id="c", rows=chunk()[:2],
                      control_hz=30.0, observation_id="o", observation_epoch=time.time(),
                      decision_mode="student", audit={}, secret=SECRET,
                      execution_safe=True)
        assert e.value.code == "not_single_step"


class TestStopSignals:
    def test_flags(self):
        hit = evaluate_stop_signals(configured=["estop_engaged", "outside_workspace"],
                                    observation={"outside_workspace": True},
                                    feedback=None, tolerance_rad=None)
        assert hit == ["outside_workspace"]

    def test_unset_tolerance_is_unevaluable(self):
        fb = {"ok": True, "commanded": MEAN_STATE, "measured": MEAN_STATE}
        hit = evaluate_stop_signals(configured=["commanded_observed_disagreement"],
                                    observation={}, feedback=fb, tolerance_rad=None)
        assert hit == ["commanded_observed_disagreement:unevaluable"]

    def test_right_arm_drift_trips(self):
        m = list(MEAN_STATE); m[10] += 0.2
        fb = {"ok": True, "commanded": MEAN_STATE, "measured": m}
        hit = evaluate_stop_signals(configured=["commanded_observed_disagreement"],
                                    observation={}, feedback=fb, tolerance_rad=0.05)
        assert hit == ["commanded_observed_disagreement"]

    def test_stop_flag_stops_the_loop(self):
        rec = loop(Mode.LIVE_SHADOW, decision()).step(obs(estop_engaged=True))
        assert rec.outcome == Outcome.STOPPED.value


def test_recorded_review_replays():
    rv = RecordedReview({"ep:t000000": decision()})
    lp = YamReviewLoop(mode=Mode.REPLAY, config={}, raw_config=load_config(),
                       proposal_source=Proposals(chunk()), review_source=rv)
    assert lp.step(obs()).outcome == Outcome.OBSERVED_ONLY.value
