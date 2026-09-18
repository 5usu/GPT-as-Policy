"""Cartesian eef gating. Fail-closed at every step; nothing defaults to allowed.

eef is the mode with the LEAST local checking available, because the controller
resolves the target into joint angles and we never see them. These tests pin the
compensating defences.
"""
from __future__ import annotations

import pytest

from .contract import EEF_PREREQUISITES, eef_execution_gate
from .eef import (MAX_POSITION_DELTA_M, MAX_ROTATION_DELTA_RAD, CartesianCapability,
                  eef_eligible, inside_workspace, quat_angle_between, validate_target)

MEASURED = {"position": [0.40, 0.10, 0.90], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}


def attested(**over) -> CartesianCapability:
    kw = dict(tcp_transform={"xyz": [0, 0, 0.12]},
              workspace={"min": [0.0, -0.5, 0.2], "max": [1.0, 0.5, 1.4]},
              collision_model={"cell": "box"}, rsi_cartesian_configured=True,
              rist_readback_verified=True)
    kw.update(over)
    return CartesianCapability(**kw)


class TestNothingDefaultsToAllowed:
    def test_bare_capability_is_refused(self):
        assert CartesianCapability().allowed()[0] is False

    @pytest.mark.parametrize("field,value", [
        ("tcp_transform", None), ("workspace", None), ("collision_model", None),
        ("rsi_cartesian_configured", False), ("rist_readback_verified", False)])
    def test_each_attestation_is_individually_required(self, field, value):
        assert attested(**{field: value}).allowed()[0] is False

    def test_static_gate_still_off(self):
        allowed, missing, _ = eef_execution_gate()
        assert allowed is False and set(missing) == set(EEF_PREREQUISITES)

    def test_deployed_rsi_is_recorded_as_joint_only(self):
        """The evidence: the shipped .src has no TOOL/RIst/RKorr."""
        assert "rsi_cartesian_configured" in EEF_PREREQUISITES
        assert EEF_PREREQUISITES["rsi_cartesian_configured"] is False


class TestTargetBounding:
    def test_small_move_is_clean(self):
        t = {"position": [0.42, 0.10, 0.90], "quaternion_wxyz": [1.0, 0, 0, 0]}
        assert validate_target(t, measured=MEASURED, capability=attested()) == []

    def test_far_move_refused(self):
        t = {"position": [0.70, 0.10, 0.90], "quaternion_wxyz": [1.0, 0, 0, 0]}
        codes = [v.code for v in validate_target(t, measured=MEASURED,
                                                 capability=attested())]
        assert "position_delta" in codes

    def test_bound_matches_upstream(self):
        assert MAX_POSITION_DELTA_M == 0.05 and MAX_ROTATION_DELTA_RAD == 0.35

    def test_large_rotation_refused(self):
        import math
        half = math.pi / 2
        t = {"position": [0.40, 0.10, 0.90],
             "quaternion_wxyz": [math.cos(half / 2), 0, 0, math.sin(half / 2)]}
        codes = [v.code for v in validate_target(t, measured=MEASURED,
                                                 capability=attested())]
        assert "rotation_delta" in codes

    def test_outside_workspace_refused(self):
        t = {"position": [0.42, 0.10, 2.90], "quaternion_wxyz": [1.0, 0, 0, 0]}
        codes = [v.code for v in validate_target(t, measured=MEASURED,
                                                 capability=attested())]
        assert "outside_workspace" in codes

    def test_unparseable_workspace_is_not_inside(self):
        assert inside_workspace([0.4, 0.1, 0.9], {"garbage": True}) is False
        assert inside_workspace([0.4, 0.1, 0.9], None) is False

    def test_no_measured_pose_refuses(self):
        t = {"position": [0.42, 0.10, 0.90], "quaternion_wxyz": [1.0, 0, 0, 0]}
        codes = [v.code for v in validate_target(t, measured={},
                                                 capability=attested())]
        assert codes == ["no_measured_pose"]

    def test_non_finite_refused(self):
        t = {"position": [float("nan"), 0.1, 0.9], "quaternion_wxyz": [1.0, 0, 0, 0]}
        assert [v.code for v in validate_target(t, measured=MEASURED,
                                                capability=attested())] == ["non_finite"]

    def test_degenerate_quaternion_refused(self):
        t = {"position": [0.42, 0.1, 0.9], "quaternion_wxyz": [0.0, 0, 0, 0]}
        assert [v.code for v in validate_target(t, measured=MEASURED,
                                                capability=attested())] == \
            ["degenerate_quaternion"]

    def test_malformed_shapes_refused(self):
        for t in ({"position": [0.4, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
                  {"position": [0.4, 0.1, 0.9], "quaternion_wxyz": [1, 0, 0]}):
            assert validate_target(t, measured=MEASURED, capability=attested())

    def test_unattested_cell_short_circuits(self):
        """It must not pretend to check a target it cannot bound."""
        t = {"position": [99.0, 99.0, 99.0], "quaternion_wxyz": [1.0, 0, 0, 0]}
        v = validate_target(t, measured=MEASURED, capability=CartesianCapability())
        assert [x.code for x in v] == ["eef_not_capable"]

    def test_eligibility_is_fail_closed(self):
        from .eef import EefViolation
        assert eef_eligible([])[0] is True
        assert eef_eligible([EefViolation("x", "y")])[0] is False


class TestQuatAngle:
    def test_identical_is_zero(self):
        assert quat_angle_between([1, 0, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.0)

    def test_sign_flip_is_zero(self):
        """q and -q are the same rotation."""
        assert quat_angle_between([1, 0, 0, 0], [-1, 0, 0, 0]) == pytest.approx(0.0)

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError):
            quat_angle_between([0, 0, 0, 0], [1, 0, 0, 0])


class TestEefThroughTheLoop:
    """The two gates are layered: static build gate, then per-cell attestation."""

    def _loop(self, cartesian=None, measured=None):
        import time
        from .loop import KukaReviewLoop, Outcome
        from .safety import CommandLedger, Mode, Supervisor
        from .test_kuka_gates import (ROBOT, SECRET, STATE, armable_raw_config,
                                      chunks, decisions, full_config)
        from .transports import FakeKukaGateway, OfflineProposalSource, RecordedReview
        lp = KukaReviewLoop(
            mode=Mode.REVIEWED_EXECUTION, config=full_config(),
            raw_config=armable_raw_config(),
            proposal_source=OfflineProposalSource(chunks()),
            review_source=RecordedReview({"obs-feasible": decisions()["obs-eef"]}),
            gateway=FakeKukaGateway(SECRET), target=ROBOT, allowlist=[ROBOT],
            supervisor=Supervisor(time.time(), True, True),
            ledger=CommandLedger(), secret=SECRET, cartesian=cartesian)
        obs = {"observation_id": "obs-feasible", "state": list(STATE),
               "epoch": time.time()}
        if measured is not None:
            obs["measured_pose"] = measured
        return lp, lp.step(obs), lp.gateway

    def test_eef_refused_and_nothing_sent(self):
        from .loop import Outcome
        _lp, rec, gw = self._loop()
        assert rec.outcome == Outcome.EEF_REFUSED.value
        assert gw.received == []
        assert rec.envelope is None

    def test_decision_is_recorded_not_discarded(self):
        """Upstream compatibility: eef is accepted as a decision, then refused."""
        _lp, rec, _gw = self._loop()
        g = rec.decision["eef_gate"]
        assert g["accepted_as_upstream_decision"] is True
        assert g["execution_allowed"] is False
        assert rec.decision["mode"] == "eef"

    def test_both_layers_reported_separately(self):
        _lp, rec, _gw = self._loop(cartesian=attested(), measured=MEASURED)
        g = rec.decision["eef_gate"]
        assert g["cell_attested"] is True, "the cell attested everything"
        assert g["static_gate_allowed"] is False, "the build gate is still shut"
        assert g["execution_allowed"] is False, "either one shutting is enough"

    def test_resolver_caveat_is_in_the_record(self):
        _lp, rec, _gw = self._loop()
        assert "CONTROLLER" in rec.decision["eef_gate"]["resolver_note"]
