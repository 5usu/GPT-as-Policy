"""The Qwen -> Astra routing on YAM. Mock monitor only; no model, no paid call."""
from __future__ import annotations


from ..kuka.vlm_backends import MockMonitorBackend
from ..kuka.vlm_monitor import Disposition
from .conftest import MEAN_STATE, chunk
from .contract import CHUNK_STEPS
from .pipeline import (Event, PolicyMode, YamEventDetector, YamMonitorSchedule,
                       YamPolicyPipeline, describe_trajectory, state_text)


def fast():
    return YamMonitorSchedule(min_interval_s=1e-6, max_interval_s=1e-6,
                              measured_latency_s=1e-6)


def reading(**over):
    r = {"phase": "approach", "progress": "normal", "target_visible": True,
         "grasp_confirmed": False, "slip_detected": False, "intent": "aligned",
         "confidence": 0.9, "execute_steps": 5, "escalate": False,
         "evidence": "both grippers approaching the blocks"}
    r.update(over)
    return r


ADVERSE = reading(progress="failed", evidence="left block dropped")


def with_grip(left, right, base=MEAN_STATE):
    s = list(base); s[6], s[13] = left, right
    return s


class TestDescribe:
    def test_both_arms_and_values(self):
        t = describe_trajectory(chunk(), MEAN_STATE)
        assert "for two arms" in t and " L [" in t and " R [" in t
        assert "left net joint displacement" in t and "right net" in t

    def test_closing_is_a_decrease_on_yam(self):
        rows = chunk(grip=(0.9, 0.9))
        rows[-1][6], rows[-1][13] = 0.1, 0.95
        t = describe_trajectory(rows)
        assert "left gripper 0.90 -> 0.10 (closing)" in t
        assert "right gripper held" in t

    def test_no_chunk_means_uncertain(self):
        assert "uncertain" in describe_trajectory([])

    def test_state_text_names_polarity(self):
        assert "0 closed, 1 open" in state_text(MEAN_STATE)


class TestEvents:
    def test_close_fires_when_opening_drops_below_half(self):
        d = YamEventDetector(fast())
        d.due(state=with_grip(0.9, 0.9), now=0.0)
        due, ev = d.due(state=with_grip(0.9, 0.3), now=1.0)
        assert due and ev is Event.GRIPPER_CLOSE and d.last_event_arm == "right"

    def test_rising_opening_is_an_open(self):
        d = YamEventDetector(fast())
        d.due(state=with_grip(0.2, 0.9), now=0.0)
        due, ev = d.due(state=with_grip(0.8, 0.9), now=1.0)
        assert ev is Event.GRIPPER_OPEN and d.last_event_arm == "left"

    def test_kuka_polarity_would_be_wrong_here(self):
        # rising through 0.5 is CLOSE on the KUKA; on YAM it must not be
        d = YamEventDetector(fast())
        d.due(state=with_grip(0.3, 0.3), now=0.0)
        _, ev = d.due(state=with_grip(0.7, 0.3), now=1.0)
        assert ev is not Event.GRIPPER_CLOSE

    def test_stall_over_twelve_joints(self):
        sched = YamMonitorSchedule(min_interval_s=1e-6, max_interval_s=100.0,
                                   stall_cycles=3, measured_latency_s=1e-6)
        d = YamEventDetector(sched)
        evs = [d.due(state=MEAN_STATE, now=float(i))[1] for i in range(6)]
        assert Event.STALLED_MOTION in evs

    def test_moving_right_arm_only_is_not_a_stall(self):
        sched = YamMonitorSchedule(min_interval_s=1e-6, max_interval_s=100.0,
                                   stall_cycles=3, measured_latency_s=1e-6)
        d = YamEventDetector(sched)
        evs = []
        for i in range(8):
            s = list(MEAN_STATE); s[9] += 0.01 * i
            evs.append(d.due(state=s, now=float(i))[1])
        assert Event.STALLED_MOTION not in evs

    def test_schedule_log_uses_chunk_clock(self):
        log = YamMonitorSchedule(measured_latency_s=6.0).to_log()
        assert log["control_cycles_between_looks"] == 180
        assert "stall_epsilon_deg" not in log


class TestPipeline:
    def make(self, readings, *, shadow=True, astra=None,
             mode=PolicyMode.PI05_LOCAL_MONITOR_ASTRA):
        return YamPolicyPipeline(mode, backend=MockMonitorBackend(readings),
                                 schedule=fast(), shadow=shadow, astra_review=astra)

    def step(self, p, i, **kw):
        return p.step(state=MEAN_STATE, proposed_steps=CHUNK_STEPS,
                      frames=[("t top", "data:image/jpeg;base64,AA")],
                      proposed_chunk=chunk(), task="stack the blocks",
                      episode_id="ep", now=float(i), **kw)

    def test_pi05_only_never_asks_the_monitor(self):
        p = YamPolicyPipeline(PolicyMode.PI05_ONLY, backend=MockMonitorBackend([]))
        out = p.step(state=MEAN_STATE, proposed_steps=16)
        assert out.controller == "pi05" and out.gate is None

    def test_shadow_keeps_the_baseline(self):
        p = self.make([reading(progress="stalled", evidence="no motion")])
        out = self.step(p, 1)
        assert out.shadow and out.executed_steps == out.baseline_steps

    def test_chunk_clamp_is_sixteen(self):
        p = self.make([reading(execute_steps=40)], shadow=False)
        assert p.gate.chunk_steps == CHUNK_STEPS
        out = self.step(p, 1)
        assert out.executed_steps <= 8

    def test_escalates_once_with_a_bimanual_packet(self):
        seen = []

        def astra(pkt):
            seen.append(pkt)
            return {"ok": True, "decision": {"mode": "student", "steps": 3}}

        p = self.make([ADVERSE] * 5, shadow=False, astra=astra)
        outs = [self.step(p, i) for i in range(1, 6)]
        assert [o.astra_called for o in outs] == [False, False, True, False, False]
        assert outs[2].controller == "astra" and outs[2].executed_steps == 3
        pkt = seen[0]
        assert pkt["schema"].endswith("yam.packet.v1")
        assert "LEFT and RIGHT" in pkt["system"]
        assert pkt["escalation"]["adverse_streak"] == 3
        assert set(pkt["response_schema"]["properties"]["edit"]["properties"]) == {"left", "right"}

    def test_astra_stop_means_zero_steps(self):
        p = self.make([ADVERSE] * 3, shadow=False,
                      astra=lambda pkt: {"ok": True, "decision": {"mode": "stop"}})
        outs = [self.step(p, i) for i in range(1, 4)]
        assert outs[-1].executed_steps == 0

    def test_hand_back_after_recovery(self):
        p = self.make([ADVERSE] * 3 + [reading()] * 3, shadow=False,
                      astra=lambda pkt: {"ok": True, "decision": {"mode": "student", "steps": 2}})
        outs = [self.step(p, i) for i in range(1, 7)]
        assert any(o.handed_back for o in outs) and outs[-1].controller == "pi05"

    def test_local_monitor_mode_never_escalates(self):
        called = []
        p = self.make([ADVERSE] * 4, shadow=False, astra=called.append,
                      mode=PolicyMode.PI05_LOCAL_MONITOR)
        outs = [self.step(p, i) for i in range(1, 5)]
        assert not called and not any(o.astra_called for o in outs)
        assert outs[2].gate["disposition"] == Disposition.ESCALATE.value

    def test_no_reading_is_hold(self):
        p = self.make([], shadow=False)
        out = self.step(p, 1)
        assert out.executed_steps == 0 and out.gate["disposition"] == "fail_safe"

    def test_metrics_tagged(self):
        p = self.make([reading()])
        self.step(p, 1)
        m = p.metrics()
        assert m["robot"] == "yam_bimanual" and m["cycles"] == 1
