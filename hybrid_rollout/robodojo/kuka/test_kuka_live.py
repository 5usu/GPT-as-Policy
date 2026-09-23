"""Live observation mode: the three-model loop must not be able to move the arm.

These tests exist because this mode is the one that touches the real RSI
socket. Everything asserted here is a safety property, not a feature.
"""
import json
import time

import pytest

from .execution import ExecutionController, ProtocolAdapter, State
from .live import (RSI_HEALTHY_HZ, LiveObservationRun, ModelCycle,
                   MotionAttempted, SharedState, _rows_of, assert_cannot_move)

START = [-76.55, -94.75, 66.60, 8.53, 19.30, 5.19]
CHUNK = [[v + i * 0.01 for v in START] + [0.0] for i in range(50)]


class FakeTransport:
    def __init__(self, joints=None):
        self.joints = list(joints or START)
        self.ipoc = 0
        self.sent: list[str] = []

    def receive(self, timeout_s):
        self.ipoc += 1
        ak = " ".join(f'A{i+1}="{v:.4f}"' for i, v in enumerate(self.joints))
        return ((f'<Rob Type="KUKA">\r\n<AIPos {ak}/>\r\n'
                 f'<IPOC>{self.ipoc}</IPOC>\r\n</Rob>').encode("ascii"),
                ("127.0.0.1", 59152))

    def send(self, payload, peer):
        self.sent.append(payload.decode())

    def close(self):
        pass


class StubPipeline:
    def __init__(self):
        self.calls = 0

    def step(self, **kw):
        self.calls += 1
        return type("O", (), {"controller": "pi05", "event": None,
                              "escalated": False, "astra_called": False})()


def controller(tr, *, allow_motion=False):
    return ExecutionController(ProtocolAdapter(tr), allow_motion=allow_motion)


def FRAMES():
    """A camera that returns one frame, keyed by name as pi0.5 needs it."""
    url = "data:image/jpeg;base64,AAAA"
    return [url], {"base": {"live": True}}, {"base": url}


def run_for(run, n):
    for _ in range(n):
        run.serve_one()


class TestMotionIsStructurallyImpossible:

    def test_a_motion_enabled_controller_is_refused_at_construction(self):
        tr = FakeTransport()
        with pytest.raises(MotionAttempted) as e:
            LiveObservationRun(controller(tr, allow_motion=True),
                               pi05_infer=lambda o: CHUNK,
                               pipeline=StubPipeline())
        assert "no measured Ruckig limits" in str(e.value)

    def test_every_reply_holds_the_measured_pose_with_the_stop_flag(self):
        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run_for(run, 20)
        assert len(tr.sent) == 20, "every frame must be answered"
        for frame in tr.sent:
            assert "<Stopflag>1</Stopflag>" in frame
            assert f'A1="{START[0]:.2f}' in frame, "reply must be the MEASURED pose"

    def test_nothing_is_ever_queued_for_execution(self):
        tr = FakeTransport()
        c = controller(tr)
        run = LiveObservationRun(c, pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run_for(run, 30)
        assert not c._queue, "a trajectory was queued in an observation-only mode"
        assert c.state is not State.EXECUTING

    def test_the_audit_row_states_the_invariant_on_every_line(self, tmp_path):
        path = tmp_path / "a.jsonl"
        rec = ModelCycle(cycle=1, started_at=0.0, state_age_s=0.0)
        run = LiveObservationRun(controller(FakeTransport()),
                                 pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline(),
                                 audit_path=str(path))
        run._write(rec)
        row = json.loads(path.read_text().strip())
        assert row["sent_to_robot"] is False
        assert row["schema"] == "kuka.live.v1"


class TestTheModelLoopCannotStarveTheControlLoop:
    """The question this mode exists to answer."""

    def test_rsi_rate_while_thinking_is_measured_not_assumed(self):
        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run._thinking.set()
        run.thinking_seconds = 1.0
        run_for(run, 240)
        s = run.summary(elapsed_s=1.0)
        assert s["rsi_rate_while_thinking"] == 240.0
        assert s["rsi_healthy_while_thinking"] is True

    def test_a_starved_control_loop_is_called_out_as_unsafe(self):
        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run._thinking.set()
        run.thinking_seconds = 1.0
        run_for(run, 50)                       # 50 Hz -- far below 250
        s = run.summary(elapsed_s=1.0)
        assert s["rsi_healthy_while_thinking"] is False
        assert "NOT yet safe" in s["verdict"]

    def test_a_healthy_run_still_refuses_to_call_itself_ready(self):
        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run._thinking.set()
        run.thinking_seconds = 1.0
        run_for(run, 250)
        v = run.summary(elapsed_s=1.0)["verdict"]
        assert "necessary, not sufficient" in v
        assert "Ruckig" in v and "deviation monitor" in v

    def test_no_frames_is_reported_as_nothing_tested(self):
        run = LiveObservationRun(controller(FakeTransport()),
                                 pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        assert "nothing about" in run.summary(elapsed_s=1.0)["verdict"]


class TestSharedStateIsReadOnlyToTheModels:

    def test_a_snapshot_is_a_copy_not_a_reference(self):
        st = SharedState()
        st.publish(START, 7, time.time())
        joints, ipoc, _ = st.snapshot()
        joints[0] = 999.0
        again, _, _ = st.snapshot()
        assert again[0] != 999.0, "the model side must not mutate measured state"
        assert ipoc == 7

    def test_state_age_is_recorded_so_a_stale_observation_is_visible(self):
        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=StubPipeline())
        run_for(run, 1)
        _, _, updated = run.shared.snapshot()
        assert updated > 0


class TestProposalParsing:

    @pytest.mark.parametrize("payload", [
        CHUNK,
        {"chunk": CHUNK},
        {"ok": True, "actions": CHUNK},
    ])
    def test_accepts_the_shapes_a_server_actually_returns(self, payload):
        assert len(_rows_of(payload)) == 50

    @pytest.mark.parametrize("payload", [
        None, {}, {"ok": False, "error": "boom"}, [], "nope", {"chunk": "nope"},
    ])
    def test_refuses_anything_it_cannot_read_as_a_chunk(self, payload):
        assert _rows_of(payload) is None

    def test_a_failed_proposal_does_not_become_a_command(self):
        tr = FakeTransport()
        pipe = StubPipeline()
        run = LiveObservationRun(controller(tr),
                                 pi05_infer=lambda o: {"ok": False, "error": "x"},
                                 pipeline=pipe, min_model_interval_s=0.0,
                                 grab_frames=FRAMES)
        run.shared.publish(START, 1, time.time())
        t = run.start_models()
        for _ in range(200):                   # wait for at least one cycle
            if run.cycles:
                break
            time.sleep(0.005)
        run.stop()
        t.join(timeout=2.0)
        assert run.cycles, "the model loop never ran"
        assert run.cycles[0].would_have_commanded is None
        assert run.cycles[0].error and "no usable chunk" in run.cycles[0].error
        assert pipe.calls == 0, \
            "a refused proposal must not reach the monitor or Astra"


class TestAssertCannotMove:

    def test_guard_accepts_a_held_controller(self):
        assert_cannot_move(controller(FakeTransport()))

    def test_guard_rejects_anything_that_claims_it_may_move(self):
        with pytest.raises(MotionAttempted):
            assert_cannot_move(type("X", (), {"allow_motion": True})())


class TestBlindInferenceIsRefused:
    """pi0.5 maps frames to observation.images.<name>; a flat list is not
    enough, and no frames at all is not an observation."""

    def _run(self, grab):
        tr = FakeTransport()
        pipe = StubPipeline()
        run = LiveObservationRun(controller(tr), pi05_infer=lambda o: CHUNK,
                                 pipeline=pipe, min_model_interval_s=0.0,
                                 grab_frames=grab)
        run.shared.publish(START, 1, time.time())
        t = run.start_models()
        for _ in range(200):
            if run.cycles:
                break
            time.sleep(0.005)
        run.stop()
        t.join(timeout=2.0)
        return run, pipe

    def test_no_frames_is_refused_rather_than_inferred_from_state(self):
        run, pipe = self._run(lambda: ([], {}, {}))
        assert run.cycles and "infer blind" in (run.cycles[0].error or "")
        assert run.cycles[0].would_have_commanded is None
        assert pipe.calls == 0

    def test_named_frames_reach_the_model_keyed_by_camera(self):
        seen = {}

        def capture(obs):
            seen.update(obs.get("images") or {})
            return CHUNK

        tr = FakeTransport()
        run = LiveObservationRun(controller(tr), pi05_infer=capture,
                                 pipeline=StubPipeline(),
                                 min_model_interval_s=0.0, grab_frames=FRAMES)
        run.shared.publish(START, 1, time.time())
        t = run.start_models()
        for _ in range(200):
            if run.cycles:
                break
            time.sleep(0.005)
        run.stop()
        t.join(timeout=2.0)
        assert "base" in seen, "pi0.5 must receive frames keyed by camera name"
        assert seen["base"].startswith("data:image/jpeg")
