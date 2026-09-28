"""Live observation on YAM with fakes: arms that hold, cameras, pi0.5, monitor."""
from __future__ import annotations

import json
import time

import pytest

from ..kuka.cameras import Frame
from ..kuka.vlm_backends import MockMonitorBackend
from .conftest import MEAN_STATE, chunk
from .contract import CAMERA_NAMES
from .live import MotionAttempted, YamLiveObservationRun, assert_cannot_move, temporal_frames
from .pipeline import PolicyMode, YamMonitorSchedule, YamPolicyPipeline
from .robot import FakeArms, HeldArms, RobotRefused


def frames(i=1):
    now = time.monotonic()
    return {c: Frame(c, b"\xff\xd8jpeg" + bytes([i]), now, time.time(), i, 640, 360)
            for c in CAMERA_NAMES}


def reading():
    return {"phase": "approach", "progress": "normal", "target_visible": True,
            "grasp_confirmed": False, "slip_detected": False, "intent": "aligned",
            "confidence": 0.8, "execute_steps": 4, "escalate": False,
            "evidence": "blocks in view"}


def pipeline(n=5):
    return YamPolicyPipeline(
        PolicyMode.PI05_LOCAL_MONITOR, backend=MockMonitorBackend([reading()] * n),
        schedule=YamMonitorSchedule(min_interval_s=1e-6, max_interval_s=1e-6,
                                    measured_latency_s=1e-6))


class TestNoMotionPath:
    def test_refuses_an_arm_that_can_move(self):
        class Mover:
            can_move_robot = False

            def command_joint_pos(self, q):
                pass
        with pytest.raises(MotionAttempted):
            assert_cannot_move(Mover())

    def test_refuses_can_move_flag(self):
        with pytest.raises(MotionAttempted):
            assert_cannot_move(FakeArms(can_move_robot=True))

    def test_held_arms_has_no_command_method(self):
        assert not any(hasattr(HeldArms, m) for m in ("command_joint_pos", "send", "command"))
        assert HeldArms.can_move_robot is False


class TestCycle:
    def run(self, infer, grab=frames, tmp_path=None):
        r = YamLiveObservationRun(FakeArms(pose=list(MEAN_STATE)), pi05_infer=infer,
                                  pipeline=pipeline(), grab_frames=grab,
                                  audit_path=str(tmp_path / "live.jsonl") if tmp_path else None,
                                  task="stack the blocks")
        r.read_one()
        return r

    def test_full_cycle(self, tmp_path):
        seen = []

        def infer(o):
            seen.append(o)
            return {"ok": True, "rows": chunk()}
        r = self.run(infer, tmp_path=tmp_path)
        rec = r.run_cycle(1)
        assert rec.error is None, rec.error
        assert rec.proposed_chunk_rows == 16 and len(rec.would_have_commanded) == 14
        o = seen[0]
        assert set(o["images"]) == set(CAMERA_NAMES) and len(o["state"]) == 14
        row = json.loads((tmp_path / "live.jsonl").read_text().splitlines()[0])
        assert row["sent_to_robot"] is False and row["robot"] == "yam_bimanual"
        assert r.summary(1.0)["commands_sent_to_robot"] == 0

    def test_missing_camera_is_blind_not_inferred(self):
        called = []
        r = self.run(lambda o: called.append(o),
                     grab=lambda: {k: v for k, v in frames().items() if k != "left"})
        rec = r.run_cycle(1)
        assert "left" in rec.error and not called

    def test_bad_chunk_is_an_error(self):
        r = self.run(lambda o: {"ok": False, "error": "contract mismatch"})
        assert "contract mismatch" in r.run_cycle(1).error

    def test_temporal_pairs_keep_one_camera_first(self):
        labelled, meta, named = temporal_frames(frames(2), frames(1))
        assert [l for l, _ in labelled[:2]] == ["t-1 top", "t top"]
        assert set(named) == set(CAMERA_NAMES) and set(meta) == set(CAMERA_NAMES)

    def test_policy_sees_commanded_gripper(self):
        class Drifting(FakeArms):
            def read(self):
                st = super().read()
                st.policy_state[6] = 0.9        # what HeldArms substitutes
                return st
        seen = []
        r = YamLiveObservationRun(Drifting(pose=list(MEAN_STATE)),
                                  pi05_infer=lambda o: seen.append(o) or {"ok": True, "rows": chunk()},
                                  pipeline=pipeline(), grab_frames=frames)
        r.read_one()
        r.run_cycle(1)
        assert seen[0]["state"][6] == 0.9


class FakeI2RT:
    def __init__(self, pos):
        self.pos, self.closed = list(pos), False

    def get_joint_pos(self):
        return list(self.pos)

    def close(self):
        self.closed = True


class TestHeldArms:
    def test_refuses_without_gripper_calibration(self):
        with pytest.raises(RobotRefused) as e:
            HeldArms(left_can="a", right_can="b", gripper_limits={"left": [0, 1]})
        assert "calibration sweep" in str(e.value)

    def test_refuses_one_channel_for_both(self):
        with pytest.raises(RobotRefused):
            HeldArms(left_can="a", right_can="a",
                     gripper_limits={"left": [0, 1], "right": [0, 1]})

    def test_reads_both_arms_and_holds_commanded_gripper(self):
        made = {}

        def factory(ch, lim):
            made[ch] = FakeI2RT(MEAN_STATE[:7] if ch == "L" else MEAN_STATE[7:])
            return made[ch]
        arms = HeldArms(left_can="L", right_can="R",
                        gripper_limits={"left": [0, 1], "right": [0, 1]},
                        factory=factory).open()
        made["L"].pos[6] = 0.2          # encoder moves; the commanded opening did not
        st = arms.read()
        assert st.measured[6] == 0.2 and st.policy_state[6] == pytest.approx(MEAN_STATE[6])
        assert st.measured[7:] == pytest.approx(MEAN_STATE[7:])
        arms.close()
        assert made["L"].closed and made["R"].closed
        with pytest.raises(RobotRefused):
            arms.read()

    def test_half_open_rig_is_closed_again(self):
        def factory(ch, lim):
            if ch == "R":
                raise OSError("no such device")
            return FakeI2RT(MEAN_STATE[:7])
        arms = HeldArms(left_can="L", right_can="R",
                        gripper_limits={"left": [0, 1], "right": [0, 1]}, factory=factory)
        with pytest.raises(OSError):
            arms.open()
        assert arms._robots == {}
