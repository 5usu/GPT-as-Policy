"""RSI gateway: framing, IPOC echo, hold/command, interpolation, refusals.

Exercised against a loopback FakeController. No robot, no API, no simulator.
"""
from __future__ import annotations

import socket
import time

import pytest

from .contract import ARM_DIM, POSITION_LIMIT_DEG
from .rsi_gateway import (FakeController, RSIGateway, RuckigInterpolator,
                          build_frame, interpolate, parse_frame)
from .safety import ArmingRefused, CommandEnvelope

SECRET = b"test-only"
START = [10.0, -20.0, 30.0, 5.0, 4.0, 3.0]


def envelope(row, secret=SECRET, approved=True):
    e = CommandEnvelope("cmd-1", "reviewed_execution", "robot", [list(row)], 1,
                        30.0, time.time(), "obs", 0.0, "student")
    if secret:
        e.sign(secret)
    e.approved_for_execution = approved
    return e


#: A deployable-shaped interpolator for tests. The real cell uses
#: Ruckig(NUM_JOINTS, 0.004); here the generator is the linear bridge, which is
#: adequate for exercising the gateway's framing and gating but is NOT the
#: deployed motion profile. Declared through RuckigInterpolator so the gateway's
#: deployability check is satisfied explicitly rather than bypassed.
def make_interpolator():
    return RuckigInterpolator(generator=interpolate)


def pair(enable_motion=False):
    """A gateway bound on loopback plus a controller pointed at it."""
    g = RSIGateway(host="127.0.0.1", port=0, enable_motion=enable_motion,
                   secret=SECRET, interpolator=make_interpolator())
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(0.5)
    g._sock = s
    return g, FakeController(s.getsockname(), start=list(START))


class TestFraming:
    def test_roundtrip_parse(self):
        xml = build_frame(42, START, gripper_pos=6000, stop_flag=1)
        ipoc, _ = parse_frame(xml)
        assert ipoc == 42

    def test_aipos_scoped_so_akorr_cannot_match(self):
        """The deployed stack hit this bug: other tags share the attribute names."""
        xml = ('<Rob><AKorr A1="9" A2="9" A3="9" A4="9" A5="9" A6="9"/>'
               '<AIPos A1="1" A2="2" A3="3" A4="4" A5="5" A6="6"/>'
               '<IPOC>5</IPOC></Rob>')
        _, joints = parse_frame(xml)
        assert joints == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_malformed_yields_none_not_a_guess(self):
        assert parse_frame("<Rob>garbage</Rob>") == (None, None)

    def test_missing_ipoc_is_none(self):
        ipoc, _ = parse_frame('<Rob><AIPos A1="1" A2="2" A3="3" A4="4" A5="5" A6="6"/></Rob>')
        assert ipoc is None


class TestInterpolation:
    def test_bridges_30hz_to_250hz(self):
        assert RSIGateway(host="127.0.0.1").cycles_per_step == 8

    def test_ends_exactly_on_target(self):
        out = interpolate([0] * 6, [8] * 6, 8)
        assert out[-1] == [8.0] * 6 and len(out) == 8

    def test_monotone_and_bounded(self):
        out = interpolate([0] * 6, [8] * 6, 8)
        col = [r[0] for r in out]
        assert col == sorted(col) and max(col) <= 8.0

    def test_single_cycle_allowed_but_is_the_full_step(self):
        assert interpolate([0] * 6, [8] * 6, 1) == [[8.0] * 6]

    def test_zero_cycles_refused(self):
        with pytest.raises(ValueError):
            interpolate([0] * 6, [1] * 6, 0)


class TestHoldMode:
    """PROPERTY: a gateway with no command still answers, holding position."""

    def test_replies_to_every_frame(self):
        g, c = pair()
        try:
            for _ in range(5):
                r = c.exchange(g)
                assert r is not None, "controller must always get a reply"
            assert g.stats.frames_out == 5
        finally:
            c.close(); g.close()

    def test_holds_measured_position(self):
        g, c = pair()
        try:
            for _ in range(3):
                c.exchange(g)
            assert [round(v, 3) for v in c.joints] == [round(v, 3) for v in START]
            assert g.stats.commanded_cycles == 0 and g.stats.hold_cycles == 3
        finally:
            c.close(); g.close()

    def test_ipoc_is_echoed_exactly(self):
        g, c = pair()
        try:
            for _ in range(4):
                r = c.exchange(g)
                assert r["echoed_ipoc"] == r["sent_ipoc"]
        finally:
            c.close(); g.close()


class TestCommandRefusals:
    """PROPERTY: motion requires every condition; any gap refuses."""

    def test_default_gateway_refuses_to_move(self):
        g, c = pair(enable_motion=False)
        try:
            c.exchange(g)
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(START + [0.5]))
            assert e.value.code == "motion_not_enabled"
        finally:
            c.close(); g.close()

    def test_refuses_before_any_session(self):
        g, _c = pair(enable_motion=True)
        try:
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(START + [0.5]))
            assert e.value.code == "no_session"
        finally:
            _c.close(); g.close()

    def test_refuses_unsigned(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(START + [0.5], secret=None))
            assert e.value.code == "bad_signature"
        finally:
            c.close(); g.close()

    def test_refuses_unapproved(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(START + [0.5], approved=False))
            assert e.value.code == "unapproved_envelope"
        finally:
            c.close(); g.close()

    def test_refuses_out_of_limit_target(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            bad = list(START); bad[1] = POSITION_LIMIT_DEG[1][1] + 5.0
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(bad + [0.5]))
            assert e.value.code == "position_limit"
        finally:
            c.close(); g.close()

    def test_refuses_multi_step(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            e = CommandEnvelope("c", "reviewed_execution", "r",
                                [START + [0.5], START + [0.5]], 2, 30.0,
                                time.time(), "o", 0.0, "student")
            e.sign(SECRET); e.approved_for_execution = True
            with pytest.raises(ArmingRefused) as exc:
                g.send(e)
            assert exc.value.code == "not_single_step"
        finally:
            c.close(); g.close()


class TestCommandedMotion:
    """PROPERTY: one authorised step moves over exactly cycles_per_step, then holds."""

    def test_one_step_is_spread_then_returns_to_hold(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            target = [v + 1.0 for v in START]
            res = g.send(envelope(target + [1.0]))
            assert res["queued_cycles"] == 8
            for _ in range(8):
                c.exchange(g)
            assert [round(v, 2) for v in c.joints] == [round(v, 2) for v in target]
            before = g.stats.hold_cycles
            c.exchange(g)
            assert g.stats.hold_cycles == before + 1, "must fall back to HOLD"
            assert g.stats.commanded_cycles == 8
        finally:
            c.close(); g.close()

    def test_no_cycle_exceeds_the_per_cycle_share(self):
        """The reason for interpolating: one 30 Hz step in one 4 ms cycle is 8x."""
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            target = [v + 4.0 for v in START]
            g.send(envelope(target + [1.0]))
            prev = list(c.joints)
            for _ in range(8):
                c.exchange(g)
                step = max(abs(a - b) for a, b in zip(c.joints, prev))
                assert step <= 4.0 / 8 + 1e-6
                prev = list(c.joints)
        finally:
            c.close(); g.close()

    def test_gripper_is_scaled_to_raw(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            g.send(envelope(list(START) + [1.0]))
            c.exchange(g)
            assert "<GRIPPER_POS>12000</GRIPPER_POS>" in c.received[-1]
        finally:
            c.close(); g.close()


class TestControlledStop:
    def test_stop_latches_and_still_replies(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            g.controlled_stop("test")
            r = c.exchange(g)
            assert r is not None, "a stopped gateway must STILL answer the controller"
            assert "<Stopflag>1</Stopflag>" in c.received[-1]
            assert g.can_move_robot is False
        finally:
            c.close(); g.close()

    def test_stop_refuses_further_commands(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            g.controlled_stop("test")
            with pytest.raises(ArmingRefused) as e:
                g.send(envelope(list(START) + [0.5]))
            assert e.value.code == "gateway_stopped"
        finally:
            c.close(); g.close()

    def test_stop_discards_queued_cycles(self):
        g, c = pair(enable_motion=True)
        try:
            c.exchange(g)
            g.send(envelope([v + 1.0 for v in START] + [0.5]))
            g.controlled_stop("test")
            pos = list(c.joints)
            c.exchange(g)
            assert [round(v, 3) for v in c.joints] == [round(v, 3) for v in pos]
        finally:
            c.close(); g.close()
