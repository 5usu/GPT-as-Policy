"""Contract, schema and packet: the YAM row shape and the unchanged upstream gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..robodojo_server.gate_assessment import GATE_INSTRUCTION
from ..skill.schema import response_schema as upstream_schema
from .conftest import MEAN_STATE, chunk
from .contract import (ACTION_DIM, ACTION_NAMES, ARM_JOINT_INDICES,
                       ARM_POSITION_LIMIT_RAD, CAMERA_NAMES, CHECKPOINT,
                       DATA_ENVELOPE_Q01, DATA_ENVELOPE_Q99, GRIPPER_INDICES,
                       JOINT_INDICES, MODEL_POSITION_RANGE_RAD, arm_joint_limit,
                       eef_execution_gate, max_step_rad, split_arms)
from .packet import build_packet
from .schema import response_schema


class TestLayout:
    def test_fourteen_values_left_then_right(self):
        assert ACTION_DIM == 14
        assert ACTION_NAMES[0] == "left_joint1" and ACTION_NAMES[6] == "left_gripper"
        assert ACTION_NAMES[7] == "right_joint1" and ACTION_NAMES[13] == "right_gripper"
        assert GRIPPER_INDICES == {"left": 6, "right": 13}
        assert set(JOINT_INDICES) | set(GRIPPER_INDICES.values()) == set(range(14))

    def test_camera_order_is_training_order(self):
        assert CAMERA_NAMES == ("top", "left", "right")

    def test_split_arms(self):
        a = split_arms(list(range(14)))
        assert a["left"]["joints"] == [0, 1, 2, 3, 4, 5] and a["left"]["gripper"] == 6
        assert a["right"]["joints"] == [7, 8, 9, 10, 11, 12] and a["right"]["gripper"] == 13

    def test_limits_are_the_driver_buffered_model_ranges(self):
        for (lo, hi), (mlo, mhi) in zip(ARM_POSITION_LIMIT_RAD, MODEL_POSITION_RANGE_RAD):
            assert lo == pytest.approx(mlo - 0.15) and hi == pytest.approx(mhi + 0.15)

    def test_training_corpus_fits_inside_the_enforced_limits(self):
        # the reason the unbuffered model range is NOT used: joint2 q01 < 0
        for j in JOINT_INDICES:
            lo, hi = arm_joint_limit(j)
            assert lo <= DATA_ENVELOPE_Q01[j] and DATA_ENVELOPE_Q99[j] <= hi
        assert DATA_ENVELOPE_Q01[8] < MODEL_POSITION_RANGE_RAD[1][0]

    def test_grippers_span_zero_to_one_in_the_corpus(self):
        for g in GRIPPER_INDICES.values():
            assert DATA_ENVELOPE_Q01[g] < 0.05 and DATA_ENVELOPE_Q99[g] > 0.95

    def test_step_cap(self):
        assert max_step_rad(30.0) == pytest.approx(2.2 / 30.0)

    def test_eef_is_refused(self):
        ok, missing, why = eef_execution_gate()
        assert not ok and "arm_base_frames_calibrated" in missing and "REFUSED" in why

    def test_experiment_config_matches_contract(self):
        import tomllib
        cfg = tomllib.loads((Path(__file__).parent / "experiments" / "bimanual_blocks"
                             / "config.toml").read_text())
        ck = cfg["checkpoint"]
        assert ck["revision"] == CHECKPOINT["revision"]
        assert ck["norm_stats_sha256"] == CHECKPOINT["norm_stats_sha256"]
        assert ck["chunk_steps"] == 16 and len(ck["action_names"]) == 14
        assert tuple(ck["action_names"]) == ACTION_NAMES
        assert tuple(ck["cameras"]) == CAMERA_NAMES


class TestSchema:
    def test_same_fields_as_upstream(self):
        ours, up = response_schema("r"), upstream_schema("r")
        assert set(ours["properties"]) == set(up["properties"])
        assert ours["properties"]["mode"] == up["properties"]["mode"]
        assert ours["properties"]["steps"] == up["properties"]["steps"]
        assert ours["properties"]["assessment"] == up["properties"]["assessment"]

    def test_left_right_wrappers_kept(self):
        s = response_schema()
        for field in ("edit", "target"):
            assert set(s["properties"][field]["properties"]) == {"left", "right"}

    def test_edit_is_joint_space_radians(self):
        arm = response_schema()["properties"]["edit"]["properties"]["left"]
        assert set(arm["properties"]) == {"delta_joint_rad", "gripper"}
        assert arm["properties"]["delta_joint_rad"]["minItems"] == 6

    def test_strict(self):
        s = response_schema()
        assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


class TestPacket:
    def pkt(self, **kw):
        base = dict(task_instruction="stack the blocks", observation_id="ep:t000000",
                    state=MEAN_STATE, chunk=chunk(), provenance="model_predicted",
                    frames={c: {"frame_index": 0} for c in CAMERA_NAMES})
        base.update(kw)
        return build_packet(**base)

    def test_gate_instruction_verbatim_first(self):
        p = self.pkt()
        assert p["system"].startswith(GATE_INSTRUCTION)
        assert p["sends_nothing"] is True

    def test_contract_note_states_polarity_and_units(self):
        s = self.pkt()["system"]
        assert "0 = CLOSED and 1 = OPEN" in s and "RADIANS" in s and "LEFT and RIGHT" in s

    def test_both_arms_rendered(self):
        t = self.pkt()["user_text"]
        assert "current_left_joints_rad" in t and "current_right_joints_rad" in t
        assert " L [" in t and " R [" in t
        assert "robocurve/pi0.5-yam" in t

    def test_unknown_provenance_refused(self):
        with pytest.raises(ValueError):
            self.pkt(provenance="guess")

    def test_response_schema_pinned_to_request(self):
        p = self.pkt()
        assert p["response_schema"]["properties"]["request_id"]["enum"] == ["ep:t000000"]
        json.dumps(p)

    def test_fk_rendered_per_arm(self):
        from .kinematics import make_fk
        fk = make_fk()
        if not fk.available:
            pytest.skip("numpy/scipy not installed")
        p = self.pkt(fk_preview=fk.preview(chunk()).to_log())
        assert "gripper-mount positions" in p["user_text"]


class TestKinematics:
    def test_two_chains_one_urdf(self):
        from .kinematics import make_fk
        fk = make_fk()
        if not fk.available:
            pytest.skip("numpy/scipy not installed")
        prev = fk.preview([MEAN_STATE])
        log = prev.to_log()
        assert log["available"] and log["common_frame"] is False
        step = log["trajectory"][0]
        assert len(step["left"]) == 3 and len(step["right"]) == 3
        # same joint values on both arms -> same pose in each arm's own frame
        same = list(MEAN_STATE[:7]) * 2
        s2 = fk.preview([same]).to_log()["trajectory"][0]
        assert s2["left"] == pytest.approx(s2["right"])

    def test_home_pose_reaches_forward(self):
        from .kinematics import make_fk
        fk = make_fk()
        if not fk.available:
            pytest.skip("numpy/scipy not installed")
        p = fk.preview([[0.0] * 14]).to_log()["trajectory"][0]["left"]
        assert 0.1 < sum(v * v for v in p) ** 0.5 < 0.8

    def test_ik_refused(self):
        from .kinematics import make_fk
        with pytest.raises(RuntimeError):
            make_fk().target()


def test_every_row_index_is_classified():
    for arm, idx in ARM_JOINT_INDICES.items():
        assert GRIPPER_INDICES[arm] == idx[-1] + 1
