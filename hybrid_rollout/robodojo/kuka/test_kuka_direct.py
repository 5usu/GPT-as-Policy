"""Joint-space Astra-direct: reachable through this controller, hard to arm.

Direct proposal removes pi0.5 as a sanity floor, so the bounds in this module are
the only thing between a bad number and the arm. Every one is required.
"""
from __future__ import annotations

import time

import pytest

from .astra_direct import (MODE, UPSTREAM_DIVERGENCE, DirectBounds,
                           DirectBoundsMissing, direct_eligible, response_schema,
                           validate_direct)
from .contract import ARM_DIM, POSITION_LIMIT_DEG
from .loop import EXECUTABLE_DECISION_MODES, KukaReviewLoop, Outcome
from .safety import CommandLedger, Mode, Supervisor
from .test_kuka_gates import (ROBOT, SECRET, STATE, armable_raw_config, chunks,
                              full_config)
from .transports import FakeKukaGateway, OfflineProposalSource, RecordedReview

M = list(STATE[:ARM_DIM])


def bounds(**over):
    cfg = {"direct_max_step_deg": 1.0, "direct_max_total_excursion_deg": 3.0,
           "direct_max_steps": 5}
    cfg.update(over)
    return DirectBounds.from_config(cfg)


def decision(rows, mode=MODE, steps=2):
    return {"request_id": "obs-feasible", "mode": mode, "steps": steps,
            "reason": "reach toward the handle",
            "joint_targets_deg": rows,
            "assessment": {"execution_status": "progressing",
                           "intent_status": "aligned"}}


def loop(rows, *, direct_bounds, mode=MODE):
    gw = FakeKukaGateway(SECRET)
    lp = KukaReviewLoop(
        mode=Mode.ASTRA_DIRECT, config=full_config(),
        raw_config=armable_raw_config(),
        proposal_source=OfflineProposalSource(chunks()),
        review_source=RecordedReview({"obs-feasible": decision(rows, mode)}),
        gateway=gw, target=ROBOT, allowlist=[ROBOT],
        supervisor=Supervisor(time.time(), True, True), ledger=CommandLedger(),
        secret=SECRET, direct_bounds=direct_bounds)
    rec = lp.step({"observation_id": "obs-feasible", "state": list(STATE),
                   "epoch": time.time()})
    return rec, gw


class TestItIsNotUpstreamsDirect:
    def test_mode_name_is_distinct(self):
        assert MODE == "astra_direct_joint"
        assert MODE != "astra_direct" and MODE != "eef"

    def test_divergence_is_stated_not_hidden(self):
        assert "NOT comparable" in UPSTREAM_DIVERGENCE
        assert "Cartesian-only" in UPSTREAM_DIVERGENCE

    def test_schema_asks_for_joint_targets_not_a_pose(self):
        props = response_schema()["properties"]
        assert "joint_targets_deg" in props
        assert "target" not in props and "edit" not in props

    def test_only_this_mode_or_stop(self):
        assert response_schema()["properties"]["mode"]["enum"] == [MODE, "stop"]


class TestBoundsAreRequired:
    def test_no_config_refuses(self):
        with pytest.raises(DirectBoundsMissing) as e:
            DirectBounds.from_config({})
        assert "direct_max_step_deg" in e.value.missing

    @pytest.mark.parametrize("key", ["direct_max_step_deg",
                                     "direct_max_total_excursion_deg",
                                     "direct_max_steps"])
    def test_each_bound_individually_required(self, key):
        cfg = {"direct_max_step_deg": 1.0,
               "direct_max_total_excursion_deg": 3.0, "direct_max_steps": 5}
        del cfg[key]
        with pytest.raises(DirectBoundsMissing):
            DirectBounds.from_config(cfg)

    def test_non_positive_bound_refused(self):
        with pytest.raises(DirectBoundsMissing):
            bounds(direct_max_step_deg=[1, 1, 1, 1, 1, -1])

    def test_loop_refuses_without_bounds(self):
        rec, gw = loop([[v + 0.5 for v in M]], direct_bounds=None)
        assert rec.outcome == Outcome.GATE_BLOCKED.value
        assert "sanity floor" in rec.reason
        assert gw.received == []


class TestBounding:
    def test_small_move_is_clean(self):
        rows = [[M[0] + 0.5] + M[1:], [M[0] + 1.0] + M[1:]]
        assert validate_direct(rows, measured=M, bounds=bounds()) == []

    def test_single_large_step_refused(self):
        codes = [v.code for v in validate_direct([[M[0] + 5.0] + M[1:]],
                                                 measured=M, bounds=bounds())]
        assert "step_too_large" in codes

    def test_accumulated_walk_refused(self):
        """Five individually-legal steps that add up to somewhere far."""
        rows = [[M[0] + 0.9 * i] + M[1:] for i in range(1, 6)]
        codes = {v.code for v in validate_direct(rows, measured=M, bounds=bounds())}
        assert "excursion_too_large" in codes
        assert "step_too_large" not in codes, "each step alone was legal"

    def test_excursion_is_from_measured_not_from_the_previous_row(self):
        rows = [[M[0] + 3.5] + M[1:]]
        codes = {v.code for v in validate_direct(rows, measured=M,
                                                 bounds=bounds(direct_max_step_deg=10.0))}
        assert "excursion_too_large" in codes

    def test_too_many_steps_refused(self):
        rows = [[M[0] + 0.1 * i] + M[1:] for i in range(1, 9)]
        codes = {v.code for v in validate_direct(rows, measured=M, bounds=bounds())}
        assert "too_many_steps" in codes

    def test_position_limit_refused(self):
        rows = [[POSITION_LIMIT_DEG[0][1] + 5.0] + M[1:]]
        codes = {v.code for v in validate_direct(rows, measured=M,
                                                 bounds=bounds(direct_max_step_deg=1e6,
                                                               direct_max_total_excursion_deg=1e6))}
        assert "position_limit" in codes

    def test_outside_training_envelope_flagged(self):
        rows = [[0.0] + M[1:]]
        codes = {v.code for v in validate_direct(rows, measured=M,
                                                 bounds=bounds(direct_max_step_deg=1e6,
                                                               direct_max_total_excursion_deg=1e6))}
        assert "outside_training_envelope" in codes

    def test_no_measured_pose_refuses(self):
        v = validate_direct([[0.0] * 6], measured=[], bounds=bounds())
        assert [x.code for x in v] == ["no_measured_pose"]

    def test_non_finite_refused(self):
        rows = [[float("nan")] + M[1:]]
        assert any(x.code == "non_finite"
                   for x in validate_direct(rows, measured=M, bounds=bounds()))

    def test_empty_refused(self):
        assert validate_direct([], measured=M, bounds=bounds())[0].code == "empty"

    def test_eligibility_is_fail_closed(self):
        from .astra_direct import DirectViolation
        assert direct_eligible([])[0] is True
        assert direct_eligible([DirectViolation("x", "y")])[0] is False


class TestThroughTheLoop:
    def test_bounded_proposal_reaches_the_gateway(self):
        rows = [[M[0] + 0.5] + M[1:], [M[0] + 1.0] + M[1:]]
        rec, gw = loop(rows, direct_bounds=bounds())
        assert rec.emitted_mode == MODE
        assert len(gw.received) == 1, "one supervised step"
        assert gw.received[0]["n_steps"] == 1

    def test_out_of_bounds_proposal_is_blocked_and_recorded(self):
        rec, gw = loop([[M[0] + 9.0] + M[1:]], direct_bounds=bounds())
        assert rec.outcome == Outcome.GATE_BLOCKED.value
        assert gw.received == []
        g = rec.decision["direct_gate"]
        assert g["accepted"] is False and g["violations"]

    def test_bounds_are_recorded_in_the_audit(self):
        rows = [[M[0] + 0.5] + M[1:]]
        rec, _gw = loop(rows, direct_bounds=bounds())
        assert rec.decision["direct_gate"]["bounds"]["mode"] == MODE

    def test_cartesian_modes_remain_blocked(self):
        from .interfaces import cartesian_capability
        assert cartesian_capability()[0] is False
        assert "eef" not in EXECUTABLE_DECISION_MODES
        assert "astra_direct" not in EXECUTABLE_DECISION_MODES

    def test_joint_direct_is_emittable_cartesian_is_not(self):
        assert MODE in EXECUTABLE_DECISION_MODES
