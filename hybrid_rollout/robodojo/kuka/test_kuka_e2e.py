"""End-to-end with a FAKE transport. Nothing binds, nothing moves, nothing is paid for.

Chain: recorded observation -> pi0.5 proposal -> recorded/mock Astra decision ->
validation -> Ruckig interpolation -> RSI frames -> measured feedback ->
progress/replan -> success or HOLD/FAULT.

Then the ten failure scenarios, each of which must land in HOLD or FAULT while
STILL ANSWERING the controller.
"""
from __future__ import annotations

import time

import pytest

from .cameras import CameraError, Frame, LiveCameras
from .context import CycleHistory, ExecutedPrefix, FrameRef
from .deployment_config import blank, missing, report
from .execution import (ArmingRequest, ExecutionController, HoldReason,
                        ProtocolAdapter, State)
from .gripper import GripperCapability, GripperController, SafeAction
from .monitor import DeviationMonitor, ToleranceConfig, ToleranceMissing, Verdict
from .otg import JointLimits, LimitsMissing, RuckigOTG, check_profile
from .review_client import ReviewOutcome, StreamingReviewClient
from .rsi_gateway import build_frame

START = [-76.55, -94.75, 66.60, 8.53, 19.30, 5.19]
LIMITS = {"max_velocity_deg_s": [85, 40, 125, 125, 125, 320],
          "max_acceleration_deg_s2": [800, 250, 500, 1200, 1200, 2500],
          "max_jerk_deg_s3": [12000, 12000, 15000, 20000, 20000, 40000],
          "source": "test"}
TOL = {"commanded_observed_tolerance_deg": 0.5,
       "tolerance_severe_multiplier": 4.0, "tolerance_persistence_cycles": 3}


class FakeTransport:
    """A controller that sends Rob frames and records the Sen replies."""

    def __init__(self, joints=None, *, drop_every=0, freeze_ipoc=False,
                 track=True, malformed=False):
        self.joints = list(joints or START)
        self.ipoc = 0
        self.sent: list[str] = []
        self.drop_every = drop_every
        self.freeze_ipoc = freeze_ipoc
        self.track = track
        self.malformed = malformed
        self.n = 0

    def receive(self, timeout_s):
        self.n += 1
        if self.drop_every and self.n % self.drop_every == 0:
            return None                       # dropped packet
        if self.malformed:
            return b"<Rob Type=\"KUKA\"><garbage/></Rob>", ("127.0.0.1", 59152)
        if not self.freeze_ipoc:
            self.ipoc += 1
        ak = " ".join(f'A{i+1}="{v:.4f}"' for i, v in enumerate(self.joints))
        frame = (f'<Rob Type="KUKA">\r\n<AIPos {ak}/>\r\n'
                 f'<IPOC>{self.ipoc}</IPOC>\r\n</Rob>')
        return frame.encode("ascii"), ("127.0.0.1", 59152)

    def send(self, payload, peer):
        xml = payload.decode()
        self.sent.append(xml)
        if self.track:
            import re
            m = re.search(r"<AK\s+" + r"\s+".join(
                f'A{i+1}="([-0-9.eE+]+)"' for i in range(6)), xml)
            if m:
                self.joints = [float(m.group(i + 1)) for i in range(6)]

    def stopflags(self):
        import re
        return [int(re.search(r"<Stopflag>(\d)</Stopflag>", x).group(1))
                for x in self.sent]


def controller(transport, *, allow_motion=True, preflight_ok=True, tol=TOL,
               limits=LIMITS, monitor=True):
    otg = RuckigOTG(JointLimits.from_config(limits))
    mon = DeviationMonitor(ToleranceConfig.from_config(tol)) if monitor else None
    return ExecutionController(
        ProtocolAdapter(transport), monitor=mon, otg=otg,
        preflight=lambda: (preflight_ok, [] if preflight_ok else ["preflight failed"]),
        allow_motion=allow_motion, observation_max_age_s=1.0)


def arm(c):
    c.open_session()
    c.serve_cycle()                       # learn the measured pose
    return c.arm(ArmingRequest(operator="eng", confirmed_estop_reachable=True,
                               confirmed_area_clear=True))


# ---------------------------------------------------------------- happy path
class TestFullChain:
    def test_recorded_to_execution_and_back_to_hold(self):
        t = FakeTransport()
        c = controller(t)
        ok, blockers = arm(c)
        assert ok, blockers
        assert c.state is State.ARMED

        target = [v + 1.0 for v in START]
        ok, why = c.submit(target, observation_epoch=time.time())
        assert ok, why
        assert c.state is State.EXECUTING

        for _ in range(400):
            c.serve_cycle()
            if c.state is not State.EXECUTING:
                break
        assert c.state is State.ARMED, "returns to ARMED after the prefix"
        assert all(abs(t.joints[j] - target[j]) < 0.05 for j in range(6))
        assert t.sent, "the controller was answered"

    def test_every_received_frame_is_answered(self):
        t = FakeTransport()
        c = controller(t)
        arm(c)
        for _ in range(20):
            c.serve_cycle()
        assert len(t.sent) == c.adapter.frames_in

    def test_hold_answers_with_stopflag_1(self):
        t = FakeTransport()
        c = controller(t)
        c.open_session()
        for _ in range(5):
            c.serve_cycle()
        assert set(t.stopflags()) == {1}, "HOLD must answer, with STOPFLAG=1"

    def test_executing_answers_with_stopflag_0(self):
        t = FakeTransport()
        c = controller(t)
        arm(c)
        c.submit([v + 0.5 for v in START], observation_epoch=time.time())
        before = len(t.sent)
        c.serve_cycle()
        assert t.stopflags()[before] == 0

    def test_ruckig_profile_is_continuous_and_within_limits(self):
        lim = JointLimits.from_config(LIMITS)
        r = RuckigOTG(lim).generate(START, [v + 3.0 for v in START])
        assert r.finished and r.n_cycles > 1
        assert check_profile(r, lim, start=START) == []

    def test_context_gains_execution_evidence_after_a_prefix(self):
        h = CycleHistory()
        f0 = [FrameRef("base", "current", 0, "/f/b0.png")]
        c1 = h.begin(observation_id="o0", task="t", state=START + [0.0],
                     proposed_chunk=[[0.0] * 7] * 50, current_frames=f0)
        assert c1.has_execution_evidence is False
        h.commit(frames=f0, decision={"mode": "student", "steps": 5},
                 executed=ExecutedPrefix(rows=[[1.0] * 7] * 5, n_steps=5,
                                         source="student prefix"))
        c2 = h.begin(observation_id="o1", task="t", state=START + [0.0],
                     proposed_chunk=[[0.0] * 7] * 50,
                     current_frames=[FrameRef("base", "current", 5, "/f/b5.png")],
                     measured_feedback=START + [0.0])
        assert c2.has_execution_evidence is True


# ------------------------------------------------------------ failure modes
class TestFailureScenarios:
    def test_1_stale_camera_refuses_the_observation(self):
        cams = LiveCameras(mapping={"base": 0, "wrist": 2},
                           names=("base", "wrist"))
        now = time.monotonic()

        class G:
            def __init__(self, age): self.f = Frame("base", b"p", now - age, 0, 1)
            def latest(self): return self.f
        cams._grabbers = {"base": G(0.01), "wrist": G(5.0)}
        with pytest.raises(CameraError) as e:
            cams.snapshot()
        assert "old" in str(e.value)

    def test_2_wrong_camera_mapping_refused(self):
        with pytest.raises(CameraError) as e:
            LiveCameras(mapping={"base": 0, "wrist": 1}).start()
        assert "metadata" in str(e.value)

    def test_3_stale_ipoc_is_answered_but_never_commanded(self):
        """Reversed deliberately.

        The old contract refused to answer a stale IPOC. That is not a safe
        default: RSI requires a reply every cycle, so staying silent turns a
        reordered or duplicated packet into a controller fault -- the exact
        failure the reply was withheld to avoid. The real invariant is narrower:
        a stale frame must not CONSUME A COMMAND. The reply echoes that frame's
        own IPOC, so it cannot be applied to a different cycle.
        """
        t = FakeTransport(freeze_ipoc=True)
        c = controller(t)
        arm(c)
        c.submit([v + 1.0 for v in START], observation_epoch=time.time())
        answered_before = len(t.sent)
        queued_before = len(c._queue)
        for _ in range(5):
            c.serve_cycle()
        assert c.adapter.ipoc_regressions > 0
        assert len(t.sent) > answered_before, \
            "every parseable frame must be answered; silence faults the controller"
        assert len(c._queue) == queued_before, \
            "a stale frame must not advance the commanded trajectory"
        for frame in t.sent[answered_before:]:
            assert "<Stopflag>1</Stopflag>" in frame, \
                "a held reply must carry the stop flag"

    def test_3b_a_malformed_frame_still_cannot_be_answered(self):
        """The one case where silence is unavoidable: no IPOC to echo."""
        t = FakeTransport(malformed=True)
        c = controller(t)
        answered_before = len(t.sent)
        for _ in range(3):
            c.serve_cycle()
        assert c.adapter.malformed > 0
        assert len(t.sent) == answered_before, \
            "an unparseable frame has no IPOC, so guessing one is worse"

    def test_4_dropped_packet_holds_without_faulting(self):
        t = FakeTransport(drop_every=2)
        c = controller(t)
        arm(c)
        c.submit([v + 1.0 for v in START], observation_epoch=time.time())
        for _ in range(10):
            c.serve_cycle()
        assert c.state in (State.HOLD, State.ARMED, State.EXECUTING)
        assert c.state is not State.FAULT, "a dropped packet is not a fault"

    def test_5_slow_reviewer_becomes_a_hold_not_a_silent_pass(self):
        def slow(body, rid, beat):
            raise TimeoutError("The read operation timed out")
        cl = StreamingReviewClient(base_url="https://x", model="m",
                                   api_key_env="PATH", enabled=True,
                                   dry_run=False, transport=slow, max_attempts=2)
        r = cl.review({"system": "s", "user_text": "u",
                       "response_schema": {"type": "object"}})
        assert r.outcome is ReviewOutcome.HOLD_TIMEOUT
        assert r.is_hold and r.decision is None
        assert "NOT 'no objection'" in r.to_log()["hold_meaning"]

    def test_6_bad_joint_limit_refuses_to_arm(self):
        with pytest.raises(LimitsMissing) as e:
            JointLimits.from_config({"max_velocity_deg_s": [85, 40, 125, 125, 125, 320]})
        assert "max_jerk_deg_s3" in e.value.missing

    def test_6b_negative_limit_refused(self):
        with pytest.raises(LimitsMissing):
            JointLimits.from_config({**LIMITS,
                                     "max_jerk_deg_s3": [-1, 1, 1, 1, 1, 1]})

    def test_7_tolerance_breach_holds_then_faults(self):
        mon = DeviationMonitor(ToleranceConfig.from_config(TOL))
        r1 = mon.observe(commanded=[10] * 6, interpolated=[10] * 6,
                         measured=[10.7] + [10] * 5)
        assert r1.verdict is Verdict.HOLD
        for _ in range(2):
            r = mon.observe(commanded=[10] * 6, interpolated=[10] * 6,
                            measured=[10.7] + [10] * 5)
        assert r.verdict is Verdict.FAULT and mon.faulted

    def test_7b_severe_single_sample_faults_immediately(self):
        mon = DeviationMonitor(ToleranceConfig.from_config(TOL))
        r = mon.observe(commanded=[10] * 6, interpolated=[10] * 6,
                        measured=[13.0] + [10] * 5)
        assert r.verdict is Verdict.FAULT

    def test_7c_missing_tolerance_refuses_to_arm(self):
        with pytest.raises(ToleranceMissing):
            ToleranceConfig.from_config({})

    def test_8_unknown_gripper_polarity_refuses_every_command(self):
        g = GripperController(GripperCapability())
        assert g.command(1.0).emitted is False
        assert g.command(0.0).emitted is False

    def test_8b_arm_stops_while_gripper_command_active(self):
        """The scenario this module exists for."""
        cap = GripperCapability(device_id="ch340", polarity=None,
                                open_close_limits={"open": 0, "closed": 12000},
                                speed_force_limits={"force": 10},
                                safe_action=SafeAction.OPEN,
                                observed_state_ack=False, enabled=True)
        g = GripperController(cap, transport=lambda raw: None)
        g.command(1.0)                                   # asked to close
        out = g.on_arm_stop("FAULT")
        assert out.emitted is False
        assert out.refused_code == "direction_unverified"
        assert "STILL IN ITS LAST COMMANDED STATE" in out.refused_reason

    def test_8c_verified_direction_allows_the_configured_safe_action(self):
        cap = GripperCapability(device_id="ch340",
                                polarity={"open": 0, "closed": 12000},
                                open_close_limits={"open": 0, "closed": 12000},
                                speed_force_limits={"force": 10},
                                safe_action=SafeAction.OPEN,
                                observed_state_ack=True, enabled=True)
        sent = []
        g = GripperController(cap, transport=sent.append)
        g.command(1.0)
        out = g.on_arm_stop("HOLD")
        assert out.emitted is True and out.intent == "open"

    def test_9_requested_edit_is_recorded_but_not_emitted(self):
        from .loop import EDIT_EXECUTION_ENABLED, EXECUTABLE_DECISION_MODES
        assert EDIT_EXECUTION_ENABLED is False
        assert "edit" not in EXECUTABLE_DECISION_MODES

    def test_9b_cartesian_stays_blocked_by_the_interface(self):
        """Adding a joint-space direct mode must not have opened Cartesian."""
        from .interfaces import cartesian_capability
        from .loop import EXECUTABLE_DECISION_MODES
        assert cartesian_capability()[0] is False
        assert "eef" not in EXECUTABLE_DECISION_MODES
        assert "astra_direct" not in EXECUTABLE_DECISION_MODES

    def test_9c_joint_direct_still_needs_its_bounds(self):
        from .astra_direct import DirectBounds, DirectBoundsMissing
        with pytest.raises(DirectBoundsMissing):
            DirectBounds.from_config({})

    def test_10_process_error_lands_in_hold_still_answering(self):
        t = FakeTransport()
        c = controller(t)
        arm(c)

        class Boom:
            deployable = True
            def generate(self, *a, **k): raise RuntimeError("otg exploded")
        c.otg = Boom()
        ok, why = c.submit([v + 1.0 for v in START], observation_epoch=time.time())
        assert ok is False and "otg exploded" in why
        assert c.state is State.HOLD
        before = len(t.sent)
        c.serve_cycle()
        assert len(t.sent) == before + 1, "still answering after a process error"

    def test_11_stale_observation_refuses_to_submit(self):
        t = FakeTransport()
        c = controller(t)
        arm(c)
        ok, why = c.submit([v + 1.0 for v in START],
                           observation_epoch=time.time() - 60)
        assert ok is False and c.state is State.HOLD

    def test_12_fault_is_latched_and_outranks_hold(self):
        t = FakeTransport()
        c = controller(t)
        arm(c)
        c.go_fault("test fault")
        c.go_hold(HoldReason.STARTUP)
        assert c.state is State.FAULT
        c.serve_cycle()
        assert t.stopflags()[-1] == 1


# --------------------------------------------------------------- arming gates
class TestArmingGates:
    def test_motion_disabled_by_default(self):
        t = FakeTransport()
        c = controller(t, allow_motion=False)
        ok, blockers = arm(c)
        assert ok is False
        assert any("allow_motion=False" in b for b in blockers)

    def test_failed_preflight_blocks_arming(self):
        t = FakeTransport()
        c = controller(t, preflight_ok=False)
        ok, blockers = arm(c)
        assert ok is False and "preflight failed" in blockers

    def test_no_preflight_refuses_to_arm_blind(self):
        t = FakeTransport()
        c = ExecutionController(ProtocolAdapter(t), allow_motion=True)
        c.open_session()
        ok, blockers = c.arm(ArmingRequest(operator="e",
                                           confirmed_estop_reachable=True,
                                           confirmed_area_clear=True))
        assert ok is False and any("blind" in b for b in blockers)

    @pytest.mark.parametrize("field", ["confirmed_estop_reachable",
                                       "confirmed_area_clear"])
    def test_operator_confirmations_required(self, field):
        t = FakeTransport()
        c = controller(t)
        c.open_session(); c.serve_cycle()
        kw = {"operator": "e", "confirmed_estop_reachable": True,
              "confirmed_area_clear": True, field: False}
        ok, blockers = c.arm(ArmingRequest(**kw))
        assert ok is False and blockers

    def test_remote_arming_refused(self):
        req = ArmingRequest(operator="e", local=False,
                            confirmed_estop_reachable=True, confirmed_area_clear=True)
        ok, missing_ = req.valid()
        assert ok is False and any("local" in m for m in missing_)

    def test_arming_refusal_lands_in_hold(self):
        t = FakeTransport()
        c = controller(t, preflight_ok=False)
        arm(c)
        assert c.state is State.HOLD


# ------------------------------------------------------------ config gating
class TestDeploymentConfigGates:
    def test_blank_config_is_all_null(self):
        assert all(v is None for v in blank()["values"].values())

    def test_report_is_machine_readable_and_names_blockers(self):
        r = report(blank())
        assert r["overall"] == "NO-GO"
        assert r["gates"]["supervised_student_prefix"]["decision"] == "NO-GO"
        assert r["gates"]["supervised_student_prefix"]["blockers"]

    def test_recorded_replay_needs_no_measurements(self):
        assert report(blank())["gates"]["recorded_replay"]["decision"] == "GO"

    def test_shadow_needs_the_camera_mapping(self):
        assert missing(blank(), "shadow_review") == ["camera_device_mapping"]

    def test_astra_direct_requires_the_most(self):
        b = blank()
        assert len(missing(b, "astra_direct")) > len(
            missing(b, "supervised_student_prefix"))
