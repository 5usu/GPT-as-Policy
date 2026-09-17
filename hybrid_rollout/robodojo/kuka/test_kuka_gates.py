"""Real-arm gate tests. Fake A800 and fake KUKA transports only.

No real robot, API, simulator or model call. Every fixture is synthetic.

These tests are the safety argument. Each one states the property it protects,
because a gate whose purpose is undocumented gets deleted by the next person who
finds it inconvenient.
"""
from __future__ import annotations

import json
import pathlib
import time

import pytest

from .contract import MAX_CORRECTION_DEG, eef_execution_gate
from .experiment import (CameraAssessment, EpisodeManifest, assess_success,
                         flatten_config, load_config, load_manifest,
                         validate_manifest)
from .loop import FINAL_STAGE, AuditLog, KukaReviewLoop, Outcome, Stage
from .safety import (ASTRA_DIRECT_EXTRA, REQUIRED_CONFIG, ArmingRefused,
                     CommandLedger, Mode, RobotIdentity, Supervisor, authorise,
                     missing_config, require_command_capable)
from .schema import response_schema
from .transports import (FakeKukaGateway, OfflineProposalSource, RecordedReview,
                         ShadowGateway, a800_live_review_enabled)

FIX = pathlib.Path(__file__).parent / "fixtures"
SECRET = b"test-only-not-a-real-key"
STATE = [-76.55, -94.75, 66.60, 8.53, 19.30, 5.19, 0.999]
ROBOT = RobotIdentity("KUKA LBR iisy 11 R1300", "SN-TEST-0001", "172.17.255.2", 59152)


def chunks():
    return {k: v for k, v in json.loads((FIX / "chunks.json").read_text()).items()
            if not k.startswith("_")}


def decisions():
    return {k: v for k, v in json.loads((FIX / "decisions.json").read_text()).items()
            if not k.startswith("_")}


def full_config():
    """Every required value present. Values are test placeholders, and they exist
    ONLY here -- the shipped experiment config leaves them unset on purpose."""
    cfg = {k: f"test-value-for-{k}" for k in REQUIRED_CONFIG}
    cfg.update(ASTRA_DIRECT_EXTRA and {k: f"test-value-for-{k}" for k in ASTRA_DIRECT_EXTRA})
    cfg["observation_freshness_s"] = 1.0
    cfg["heartbeat_timeout_s"] = 2.0
    return cfg


def raw_config():
    return load_config()


def observation(oid="obs-feasible", epoch=None):
    return {"observation_id": oid, "state": list(STATE),
            "epoch": time.time() if epoch is None else epoch}


def armable_raw_config():
    """The shipped experiment config PLUS the one stop-condition tolerance it
    deliberately leaves unset. Supplying it here is a test placeholder, not a
    measurement -- the shipped file must stay unarmable."""
    cfg = raw_config()
    cfg["stop_conditions"] = {**cfg["stop_conditions"],
                              "commanded_observed_tolerance_deg": 0.5}
    return cfg


def loop(mode, *, gateway=None, config=None, supervisor=None, oid="obs-feasible",
         decisions_override=None, audit=None, raw=None):
    sup = supervisor or Supervisor(heartbeat_at=time.time(), deadman_held=True,
                                   estop_clear=True)
    return KukaReviewLoop(
        mode=mode, config=config if config is not None else full_config(),
        raw_config=raw if raw is not None else armable_raw_config(),
        proposal_source=OfflineProposalSource(chunks(), checkpoint_id="synthetic"),
        review_source=RecordedReview(decisions_override or decisions()),
        gateway=gateway, target=ROBOT, allowlist=[ROBOT], supervisor=sup,
        ledger=CommandLedger(), secret=SECRET, audit=audit)


# ---------------------------------------------------------------------------
class TestObservationOnlyModesCannotWrite:
    """PROPERTY: replay and live_shadow can never produce a robot command."""

    @pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.LIVE_SHADOW])
    def test_command_construction_is_unreachable(self, mode):
        with pytest.raises(ArmingRefused) as e:
            require_command_capable(mode)
        assert e.value.code == "mode_cannot_command"

    @pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.LIVE_SHADOW])
    def test_loop_stops_at_validate(self, mode):
        rec = loop(mode, gateway=FakeKukaGateway(SECRET)).step(observation())
        assert rec.stage_reached == Stage.VALIDATE.value
        assert rec.outcome == Outcome.OBSERVED_ONLY.value
        assert rec.envelope is None
        assert rec.approved_for_execution is False

    @pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.LIVE_SHADOW])
    def test_fake_kuka_receives_nothing_even_when_attached(self, mode):
        """A live gateway wired in must still receive nothing in these modes."""
        gw = FakeKukaGateway(SECRET)
        loop(mode, gateway=gw).step(observation())
        assert gw.received == []

    @pytest.mark.parametrize("mode", [Mode.REPLAY, Mode.LIVE_SHADOW])
    def test_authorise_refuses_directly(self, mode):
        with pytest.raises(ArmingRefused) as e:
            authorise(mode=mode, config=full_config(), target=ROBOT,
                      allowlist=[ROBOT],
                      supervisor=Supervisor(time.time(), True, True),
                      ledger=CommandLedger(), command_id="c1",
                      rows=[[0.0] * 7], control_hz=30.0, observation_id="o",
                      observation_epoch=time.time(), decision_mode="student",
                      audit={}, secret=SECRET, execution_safe=True)
        assert e.value.code == "mode_cannot_command"

    def test_final_stage_table_matches_capability(self):
        assert FINAL_STAGE[Mode.REPLAY] is Stage.VALIDATE
        assert FINAL_STAGE[Mode.LIVE_SHADOW] is Stage.VALIDATE


# ---------------------------------------------------------------------------
class TestMissingConfigRefusesArming:
    """PROPERTY: an unsupplied physical value blocks; nothing is invented."""

    def test_shipped_experiment_config_cannot_arm(self):
        flat = flatten_config(load_config())
        missing = missing_config(flat, Mode.REVIEWED_EXECUTION)
        assert missing, "the shipped config must NOT be armable as delivered"

    def test_every_required_key_is_individually_blocking(self):
        for key in REQUIRED_CONFIG:
            cfg = full_config(); cfg[key] = None
            assert key in missing_config(cfg, Mode.REVIEWED_EXECUTION), key

    def test_arming_names_what_is_missing(self):
        cfg = full_config(); cfg.pop("force_torque_limits")
        with pytest.raises(ArmingRefused) as e:
            authorise(mode=Mode.REVIEWED_EXECUTION, config=cfg, target=ROBOT,
                      allowlist=[ROBOT],
                      supervisor=Supervisor(time.time(), True, True),
                      ledger=CommandLedger(), command_id="c", rows=[[0.0] * 7],
                      control_hz=30.0, observation_id="o",
                      observation_epoch=time.time(), decision_mode="student",
                      audit={}, secret=SECRET, execution_safe=True)
        assert e.value.code == "missing_config"
        assert "force_torque_limits" in e.value.missing


# ---------------------------------------------------------------------------
class TestAstraDirectCannotBypassGates:
    """PROPERTY: direct mode runs the identical gate chain, plus four more."""

    def test_needs_strictly_more_config(self):
        assert set(ASTRA_DIRECT_EXTRA) - set(REQUIRED_CONFIG)
        base = {k: "v" for k in REQUIRED_CONFIG}
        assert missing_config(base, Mode.REVIEWED_EXECUTION) == []
        assert missing_config(base, Mode.ASTRA_DIRECT) == sorted(ASTRA_DIRECT_EXTRA)

    def test_cannot_arm_with_only_reviewed_execution_config(self):
        base = {k: "v" for k in REQUIRED_CONFIG}
        base["observation_freshness_s"] = 1.0; base["heartbeat_timeout_s"] = 2.0
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.ASTRA_DIRECT, gateway=gw, config=base).step(observation())
        assert rec.outcome == Outcome.ARMING_REFUSED.value
        assert rec.envelope["code"] == "missing_config"
        assert gw.received == []

    @pytest.mark.parametrize("field,value", [
        ("deadman_held", False), ("estop_clear", False)])
    def test_supervisor_gates_apply_identically(self, field, value):
        sup = Supervisor(heartbeat_at=time.time(), deadman_held=True, estop_clear=True)
        setattr(sup, field, value)
        gw = FakeKukaGateway(SECRET)
        rec = loop(Mode.ASTRA_DIRECT, gateway=gw, supervisor=sup).step(observation())
        assert rec.outcome == Outcome.ARMING_REFUSED.value
        assert gw.received == []

    def test_allowlist_applies_identically(self):
        gw = FakeKukaGateway(SECRET)
        lp = loop(Mode.ASTRA_DIRECT, gateway=gw)
        lp.allowlist = [RobotIdentity("other", "SN-X", "10.0.0.1", 1)]
        rec = lp.step(observation())
        assert rec.envelope["code"] == "not_allowlisted"
        assert gw.received == []

    def test_shipped_config_marks_direct_disabled(self):
        assert load_config()["astra_direct"]["enabled"] is False
