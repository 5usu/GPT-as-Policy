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

class TestLiveCameraCallsExist:
    """The live loop must only call methods LiveCameras actually has.

    FOUND ON THE REAL ARM: `cli live` called cams.capture(), which does not
    exist -- the class exposes snapshot(). Every model cycle raised
    AttributeError and was swallowed into the per-cycle error field, so the
    RSI loop ran at a healthy 250 Hz while 228 consecutive cycles produced no
    observation, no monitor reading and no Astra call. The run looked alive.

    A type checker would catch this; this test is the cheap equivalent, and it
    fails on the NAME rather than on behaviour so it cannot rot into a mock.
    """

    def test_every_camera_method_the_cli_calls_exists(self):
        import ast
        import inspect
        from pathlib import Path
        from .cameras import LiveCameras
        src = Path(inspect.getfile(LiveCameras)).parent / "cli.py"
        tree = ast.parse(src.read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("cams", "cameras")
        }
        missing = sorted(m for m in called if not hasattr(LiveCameras, m))
        assert not missing, (
            f"cli.py calls {missing} on LiveCameras, which has "
            f"{sorted(m for m in dir(LiveCameras) if not m.startswith('_'))}")


class TestChunkReplayInLiveMode:
    """--chunks-file must work in `cli live`, where observation ids are new.

    FOUND ON THE REAL ARM: live.py names each observation `live:{ipoc}` from the
    controller's counter, while LocalPi05ProposalSource looks chunks up BY id.
    A recorded file is keyed `<episode>:t000050`, so every lookup missed and all
    150 cycles failed with "no proposal for live:86302460". The help advertises
    the flag as "replay precomputed chunks instead of inferring", so replay must
    not depend on ids it cannot know.

    Sequential replay is opt-in: `cli run` DOES have matching ids and must keep
    its keyed lookup, where a missing chunk is a real error worth reporting.
    """

    CHUNKS = {"ep:t000000": [[1.0] * 7] * 50,
              "ep:t000050": [[2.0] * 7] * 50,
              "_meta": {"use_relative_actions": True}}

    def _src(self, **kw):
        from .transports import LocalPi05ProposalSource
        return LocalPi05ProposalSource(chunks={k: v for k, v in self.CHUNKS.items()
                                               if not k.startswith("_")}, **kw)

    def test_sequential_replay_ignores_the_id_and_serves_in_order(self):
        s = self._src(sequential=True)
        first = s.propose({"observation_id": "live:86302460"})
        second = s.propose({"observation_id": "live:86302461"})
        assert first["ok"] is True and second["ok"] is True
        assert first["rows"][0][0] == 1.0
        assert second["rows"][0][0] == 2.0

    def test_sequential_replay_cycles_rather_than_running_dry(self):
        s = self._src(sequential=True)
        served = [s.propose({"observation_id": f"live:{i}"}) for i in range(5)]
        assert all(r["ok"] for r in served), "a long session must not run out"
        assert [r["rows"][0][0] for r in served] == [1.0, 2.0, 1.0, 2.0, 1.0]

    def test_sequential_replay_says_which_recorded_chunk_it_served(self):
        """The audit must not imply this came from the live scene."""
        r = self._src(sequential=True).propose({"observation_id": "live:1"})
        assert r.get("replayed_from") == "ep:t000000"

    def test_keyed_lookup_is_unchanged_when_sequential_is_off(self):
        s = self._src()
        assert s.propose({"observation_id": "ep:t000050"})["ok"] is True
        miss = s.propose({"observation_id": "live:86302460"})
        assert miss["ok"] is False and "no proposal" in miss["error"]


class TestLiveMonitorTimeoutMatchesTheDevice:
    """`cli live` must not undercut the measured profile.

    VlmConfig.jetson() carries a MEASURED 12 s (5.1-9.1 s observed over 13
    readings at MODE_30W). cli live passed its own default of 6 s straight into
    VlmConfig.jetson(timeout_s=...), which silently overrode it -- and 6 s was
    measured failing 8 of 8 readings. A flag default must not quietly contradict
    the number the profile was measured into.
    """

    def test_live_default_is_not_below_the_measured_profile(self):
        from .cli import main as _main   # noqa: F401  (ensures the module parses)
        from .vlm_backends import VlmConfig
        import argparse, ast, inspect, pathlib
        src = pathlib.Path(inspect.getfile(VlmConfig)).parent / "cli.py"
        tree = ast.parse(src.read_text())
        found = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args and getattr(node.args[0], "value", "") == "--monitor-timeout"):
                for kw in node.keywords:
                    if kw.arg == "default":
                        found.append(ast.literal_eval(kw.value))
        assert found, "no --monitor-timeout default found to check"
        profile = VlmConfig.jetson().timeout_s
        too_low = [d for d in found if d is not None and d < profile]
        assert not too_low, (
            f"--monitor-timeout defaults {too_low} are below the measured "
            f"profile ({profile}s); 6s was measured failing every reading")
