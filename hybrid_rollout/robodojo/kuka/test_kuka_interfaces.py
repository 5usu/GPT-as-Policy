"""The three "TCP" meanings, the RSI element limit, and the execution locks.

These encode VERIFIED DEPLOYMENT FACTS. A test failing here means either the cell
changed or someone relaxed a lock; both need a human, not a fix.
"""
from __future__ import annotations

import pytest

from . import interfaces as I
from .loop import EDIT_EXECUTION_ENABLED, EXECUTABLE_DECISION_MODES
from .rsi_gateway import RSIGateway, Readiness, check_rsi_elements


class TestThreeMeaningsStaySeparate:
    def test_all_three_are_distinct(self):
        assert len({f.meaning for f in I.FACTS}) == 3

    def test_rsi_is_udp_59152_and_carries_motion(self):
        f = next(x for x in I.FACTS if x.meaning is I.Meaning.RSI_UDP)
        assert (f.port, f.protocol, f.carries_motion) == (59152, "udp", True)

    def test_ext_trigger_is_tcp_54600_and_carries_no_motion(self):
        f = next(x for x in I.FACTS if x.meaning is I.Meaning.EXT_TRIGGER_TCP)
        assert (f.port, f.protocol, f.carries_motion) == (54600, "tcp", False)

    def test_this_package_never_uses_the_trigger_channel(self):
        assert I.EXT_TRIGGER_USED_BY_THIS_PACKAGE is False

    def test_tool_center_point_has_no_port(self):
        f = next(x for x in I.FACTS if x.meaning is I.Meaning.TOOL_CENTER_POINT)
        assert f.port is None and f.protocol is None

    def test_the_two_ports_are_not_the_same(self):
        assert I.RSI_UDP_PORT != I.EXT_TRIGGER_PORT


class TestRsiElementLimit:
    def test_accepts_joints_and_stopflag_only(self):
        assert set(I.RSI_ACCEPTED_ELEMENTS) == {
            "AK.A1", "AK.A2", "AK.A3", "AK.A4", "AK.A5", "AK.A6", "STOPFLAG"}

    def test_no_rkorr(self):
        assert I.RSI_HAS_RKORR is False
        assert I.RSI_CARTESIAN_SUPPORTED is False

    def test_deployed_config_matches(self):
        assert check_rsi_elements()[0] is True

    def test_missing_element_is_a_mismatch(self):
        ok, problems = check_rsi_elements(("AK.A1", "STOPFLAG"))
        assert ok is False and "missing expected element" in problems[0]

    def test_unexpected_rkorr_is_flagged_not_embraced(self):
        ok, problems = check_rsi_elements(I.RSI_ACCEPTED_ELEMENTS + ("RKorr",))
        assert ok is False and "re-verify" in problems[0]


class TestToolCenterPoint:
    def test_tool_is_unknown_and_has_no_placeholder(self):
        assert I.TOOL_TRANSFORM_KNOWN is False
        assert I.TOOL_TRANSFORM_VALUE is None

    def test_fk_is_flange_not_tool_tip(self):
        assert I.FK_FRAME == "flange" and I.FK_IS_TOOL_TIP is False

    def test_source_json_agrees(self):
        import json
        import pathlib
        src = json.loads((pathlib.Path(__file__).parent / "SOURCE.json").read_text())
        assert "UNKNOWN" in src["robot"]["tcp_transform"]


class TestExecutionLocks:
    def test_only_student_may_be_emitted(self):
        assert EXECUTABLE_DECISION_MODES == frozenset({"student"})

    def test_edit_execution_is_off(self):
        assert EDIT_EXECUTION_ENABLED is False

    def test_cartesian_is_blocked_by_the_interface_not_only_calibration(self):
        ok, why = I.cartesian_capability()
        assert ok is False
        assert any("RKorr" in w for w in why), \
            "the interface limit must be cited, not just the missing tool"

    def test_supplying_the_tool_alone_would_not_unlock_cartesian(self):
        """Calibration is necessary but not sufficient while RKorr is absent."""
        assert I.RSI_HAS_RKORR is False
        _ok, why = I.cartesian_capability()
        assert len(why) == 2, "both blockers must be reported, not just one"

    def test_every_locked_mode_states_a_reason(self):
        for mode, reason in I.modes_locked().items():
            assert reason and len(reason) > 20, mode


class TestSafetyPosture:
    def test_hold_is_default(self):
        assert I.HOLD_IS_DEFAULT is True
        assert RSIGateway().can_move_robot is False

    def test_controlled_stop_keeps_replying(self):
        assert I.STOP_REPLIES_CONTINUE is True
        assert I.STOP_FLAG_ON_CONTROLLED_STOP == 1
        assert "silence" in I.SILENCE_IS_UNSAFE.lower()

    def test_estop_is_authoritative_and_never_written(self):
        assert "never clear or bypass" in I.ESTOP_AUTHORITATIVE
        import pathlib
        src = (pathlib.Path(__file__).parent / "rsi_gateway.py").read_text()
        assert "estop_clear = True" not in src


class TestReadinessFailsClosed:
    def test_fresh_gateway_is_not_ready(self):
        ok, blockers = RSIGateway().readiness().ready_for_hold()
        assert ok is False and blockers

    @pytest.mark.parametrize("field,value", [
        ("socket_open", False), ("listener_bound", False), ("frames_seen", 0),
        ("ipoc_monotonic", False), ("measured_pose_known", False),
        ("config_matches", False)])
    def test_each_condition_individually_blocks(self, field, value):
        r = Readiness(socket_open=True, listener_bound=True, frames_seen=3,
                      last_frame_age_s=0.004, ipoc_monotonic=True,
                      measured_pose_known=True, config_matches=True)
        setattr(r, field, value)
        assert r.ready_for_hold()[0] is False

    def test_all_conditions_met_is_ready(self):
        r = Readiness(socket_open=True, listener_bound=True, frames_seen=3,
                      last_frame_age_s=0.004, ipoc_monotonic=True,
                      measured_pose_known=True, config_matches=True)
        assert r.ready_for_hold()[0] is True

    def test_stale_frame_blocks(self):
        r = Readiness(socket_open=True, listener_bound=True, frames_seen=3,
                      last_frame_age_s=1.0, ipoc_monotonic=True,
                      measured_pose_known=True, config_matches=True)
        ok, blockers = r.ready_for_hold()
        assert ok is False and any("stale" in b for b in blockers)

    def test_latched_stop_blocks(self):
        r = Readiness(socket_open=True, listener_bound=True, frames_seen=3,
                      last_frame_age_s=0.004, ipoc_monotonic=True,
                      measured_pose_known=True, config_matches=True,
                      stopped="test")
        assert r.ready_for_hold()[0] is False

    def test_readiness_does_not_imply_motion(self):
        r = Readiness(socket_open=True, listener_bound=True, frames_seen=3,
                      last_frame_age_s=0.004, ipoc_monotonic=True,
                      measured_pose_known=True, config_matches=True)
        assert r.ready_for_hold()[0] is True
        assert r.motion_enabled is False
        assert "says nothing about whether anything may move" in r.to_log()["note"]
