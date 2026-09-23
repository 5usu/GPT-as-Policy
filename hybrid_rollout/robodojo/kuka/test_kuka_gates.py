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


class TestMatchesDeployedRobotStack:
    """PROPERTY: our numbers match the stack that actually drives the arm.

    Source of truth: KUKA/teleoperation/udp_teleoperate.py on the robot host.
    These were wrong in the first implementation -- validated against the kroshu
    URDF's mechanical limits rather than the operational limits the deployed code
    enforces, and with an RSI frame that had the wrong Type, wrong line endings,
    wrong precision and two missing elements.
    """

    def test_operational_limits_are_the_deployed_ones(self):
        from .contract import POSITION_LIMIT_DEG
        # udp_teleoperate.py:178-179, verbatim
        assert [lo for lo, _ in POSITION_LIMIT_DEG] == \
            [-184.5, -229.5, -149.5, -179.5, -109.5, -219.5]
        assert [hi for _, hi in POSITION_LIMIT_DEG] == \
            [184.5, 49.5, 149.5, 179.5, 109.5, 219.5]

    def test_operational_limits_are_inside_mechanical_limits(self):
        from .contract import POSITION_LIMIT_DEG, URDF_POSITION_LIMIT_DEG
        for (lo, hi), (mlo, mhi) in zip(POSITION_LIMIT_DEG, URDF_POSITION_LIMIT_DEG):
            assert lo > mlo and hi < mhi, "operational must be strictly inside"
            assert abs((mhi - hi) - 0.5) < 1e-9, "the deployed inset is 0.5 deg"

    def test_a_command_the_real_stack_rejects_is_rejected_here(self):
        """The bug this fixes: 184.7 deg is inside the URDF limit but outside
        what the deployed stack allows."""
        from .validation import validate_chunk
        rows = [[184.7, 0, 0, 0, 0, 0, 0.5]]
        codes = {v.code for v in validate_chunk(rows, check_envelope=False)}
        assert "position_limit" in codes

    def test_rsi_frame_matches_the_deployed_builder(self):
        from .transports import rsi_response_xml
        got = rsi_response_xml(12345, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                               gripper_pos=6000, stop_flag=0)
        expected = (
            '<Sen Type="ImFree">\r\n'
            '<AK A1="1.00" A2="2.00" A3="3.00" A4="4.00" A5="5.00" A6="6.00"/>\r\n'
            '<GRIPPER_POS>6000</GRIPPER_POS>\r\n'
            '<Stopflag>0</Stopflag>\r\n'
            '<IPOC>12345</IPOC>\r\n'
            '</Sen>')
        assert got == expected

    def test_rsi_rate_is_250hz(self):
        from .transports import RSI_CYCLE_TIME, RSI_HZ
        assert RSI_CYCLE_TIME == 0.004 and RSI_HZ == 250.0

    def test_gripper_scale_matches_the_stack(self):
        from .contract import GRIPPER_SCALE
        from .transports import gripper_to_raw
        assert GRIPPER_SCALE == 12000.0
        assert gripper_to_raw(1.0) == 12000 and gripper_to_raw(0.0) == 0
        assert gripper_to_raw(5.0) == 12000, "must clamp, not overflow"

    def test_gripper_path_is_not_assumed(self):
        """The gripper may bypass RSI entirely via Modbus RTU."""
        from .transports import GRIPPER_PATHS
        assert set(GRIPPER_PATHS) == {"rsi_gripper_pos", "direct_modbus_rtu"}

    def test_model_identification_basis_is_recorded_honestly(self):
        from .contract import DATASET_ROBOT_TYPE, MODEL_IDENTIFICATION_BASIS
        assert DATASET_ROBOT_TYPE == "kuka_lbr_iico"
        assert "never names the model" in MODEL_IDENTIFICATION_BASIS


class TestAstraBackgroundMode:
    """The proxy cuts a held connection at ~60 s; a review needs ~130 s."""

    def _src(self, submit, gets, **kw):
        import os
        from .transports import AstraReviewSource
        os.environ["BGKEY"] = "placeholder"
        self.posts = []

        def post(u, b, h, t):
            self.posts.append(b)
            return submit

        return AstraReviewSource(
            base_url="https://x/v1/responses", model="m", api_key_env="BGKEY",
            enabled=True, dry_run=False, background=True, poll_interval_s=0.001,
            transport=post, get_transport=lambda u, h, t: gets.pop(0), **kw)

    def _packet(self):
        return {"system": "s", "user_text": "u",
                "response_schema": {"type": "object"}}

    def test_submit_then_poll_returns_the_decision(self):
        s = self._src({"id": "r1", "status": "queued"},
                      [{"id": "r1", "status": "completed",
                        "output_text": '{"mode":"student"}'}])
        r = s.review(self._packet())
        assert r["ok"] and r["decision"]["mode"] == "student"
        assert r["polls"] == 1 and r["response_id"] == "r1"

    def test_an_incomplete_response_is_never_parsed_as_a_decision(self):
        """It can carry PARTIAL output; a truncated review is not a verdict."""
        s = self._src({"id": "r1", "status": "queued"},
                      [{"id": "r1", "status": "incomplete",
                        "output_text": '{"mode":"student"}',
                        "incomplete_details": {"reason": "max_output_tokens"}}])
        r = s.review(self._packet())
        assert r["ok"] is False
        assert "incomplete" in r["error"] and "partial output" in r["error"]

    def test_a_failed_review_is_a_failure_not_a_stop_decision(self):
        s = self._src({"id": "r1", "status": "queued"},
                      [{"id": "r1", "status": "failed"}])
        r = self._src({"id": "r1", "status": "queued"},
                      [{"id": "r1", "status": "failed"}]).review(self._packet())
        assert r["ok"] is False and "failed" in r["error"]

    def test_a_poll_error_retries_the_GET_and_never_resubmits(self):
        """Re-POSTing would create a second review: verdict-shopping, and a
        second charge for an answer already being computed."""
        calls = {"n": 0}

        def flaky(u, h, t):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("RemoteDisconnected")
            return {"id": "r1", "status": "completed",
                    "output_text": '{"mode":"student"}'}

        s = self._src({"id": "r1", "status": "queued"}, [])
        s.get_transport = flaky
        r = s.review(self._packet())
        assert r["ok"] is True
        assert len(self.posts) == 1, "the review must be submitted exactly once"
        assert r["polls"] == 3

    def test_the_wall_clock_deadline_is_real(self):
        """urllib's timeout is per socket operation and never bounded total
        elapsed time -- that is why a stream could hang past 180 s."""
        s = self._src({"id": "r1", "status": "queued"},
                      [], deadline_s=0.05)
        s.get_transport = lambda u, h, t: {"id": "r1", "status": "in_progress"}
        r = s.review(self._packet())
        assert r["ok"] is False and "deadline" in r["error"]
        assert "may still be running server-side" in r["error"]
        assert r["wall_clock_s"] >= 0

    def test_background_and_stream_are_mutually_exclusive(self):
        import pytest as _p
        from .transports import AstraReviewSource
        with _p.raises(ValueError) as e:
            AstraReviewSource(base_url="u", model="m", api_key_env="K",
                              background=True, stream=True)
        assert "pick one" in str(e.value)

    def test_background_forces_store_because_an_id_must_be_retrievable(self):
        from .transports import AstraReviewSource
        s = AstraReviewSource(base_url="u", model="m", api_key_env="K",
                              background=True, store=False)
        body = s.build_body({"system": "s", "user_text": "u",
                             "response_schema": {}})
        assert body["background"] is True
        assert body["store"] is True, "a backgrounded review must be retrievable"

    def test_non_background_body_is_unchanged(self):
        from .transports import AstraReviewSource
        s = AstraReviewSource(base_url="u", model="m", api_key_env="K")
        body = s.build_body({"system": "s", "user_text": "u",
                             "response_schema": {}})
        assert "background" not in body and body["store"] is False
