"""Deterministic validation and sanitization over both arms."""
from __future__ import annotations

import pytest

from .conftest import MEAN_STATE, chunk
from .contract import GRIPPER_INDICES, JOINT_INDICES, arm_joint_limit, max_step_rad
from .sanitize import sanitize, worst_velocity_ratio
from .validation import execution_eligible, has_fatal, validate_chunk, worsens


def codes(vs):
    return {(v.code, v.joint) for v in vs}


class TestValidate:
    def test_clean_chunk_is_executable(self):
        vs = validate_chunk(chunk(), state=MEAN_STATE)
        assert execution_eligible(vs) == (True, [])

    def test_bad_width_is_fatal(self):
        vs = validate_chunk([[0.0] * 7], state=MEAN_STATE)
        assert has_fatal(vs) and vs[0].code == "bad_shape"

    def test_non_finite_is_fatal(self):
        rows = chunk(); rows[3][9] = float("nan")
        assert has_fatal(validate_chunk(rows))

    @pytest.mark.parametrize("j", [1, 8])            # joint2 on each arm
    def test_position_limit_per_arm(self, j):
        rows = chunk(); lo, _ = arm_joint_limit(j)
        rows[5][j] = lo - 0.01
        vs = validate_chunk(rows, check_envelope=False)
        assert has_fatal(vs)
        name = "left_joint2" if j == 1 else "right_joint2"
        assert ("position_limit", name) in codes(vs)

    def test_right_arm_velocity_breach_is_caught(self):
        rows = chunk(step=0.0)
        rows[4][10] += max_step_rad() * 1.5
        vs = validate_chunk(rows, state=MEAN_STATE)
        assert ("velocity_limit", "right_joint4") in codes(vs)
        assert not execution_eligible(vs)[0]

    def test_state_jump(self):
        rows = chunk(step=0.0)
        s = list(MEAN_STATE); s[2] -= 0.5
        assert ("state_discontinuity", "left_joint3") in codes(validate_chunk(rows, state=s))

    def test_gripper_overshoot_is_advisory_but_blocks(self):
        rows = chunk(grip=(1.02, 0.5))
        vs = validate_chunk(rows, state=MEAN_STATE)
        assert ("gripper_out_of_range", "left_gripper") in codes(vs)
        assert not has_fatal(vs)
        assert not execution_eligible(vs)[0]

    def test_gripper_is_not_a_joint(self):
        rows = chunk(step=0.0)
        rows[1][GRIPPER_INDICES["left"]] = 0.0     # a full close in one step is fine
        rows[1][GRIPPER_INDICES["right"]] = 1.0
        assert execution_eligible(validate_chunk(rows))[0]

    def test_envelope_is_advisory(self):
        rows = chunk(step=0.0)
        for r in rows:
            r[0] = -1.5                             # outside q01, inside the limit
        vs = validate_chunk(rows)
        assert ("outside_training_envelope", "left_joint1") in codes(vs)
        assert execution_eligible(vs) == (True, [])     # a signal, not a limit

    def test_worsens(self):
        a = validate_chunk(chunk())
        rows = chunk(); rows[3][0] = arm_joint_limit(0)[1] + 1
        assert worsens(a, validate_chunk(rows))[0]


class TestSanitize:
    def test_clean_chunk_unchanged(self):
        rows = chunk()
        out, rep = sanitize(rows, state=MEAN_STATE)
        assert out == rows and rep.returned_unchanged and not rep.changed

    def test_both_grippers_clamped_joints_untouched(self):
        rows = chunk(grip=(1.03, -0.02))
        out, rep = sanitize(rows, state=MEAN_STATE)
        assert len(rep.gripper_clamps) == 2 * len(rows)
        assert all(r[6] == 1.0 and r[13] == 0.0 for r in out)
        for a, b in zip(out, rows):
            assert [a[j] for j in JOINT_INDICES] == [b[j] for j in JOINT_INDICES]

    def test_time_scaling_preserves_path(self):
        rows = chunk(step=max_step_rad() * 1.8, n=4)
        out, rep = sanitize(rows, state=MEAN_STATE)
        assert rep.time_scaling.applied and len(out) > len(rows)
        assert worst_velocity_ratio(out, 30.0)[0] <= 1.0 + 1e-9
        assert out[-1] == pytest.approx(rows[-1])
        assert execution_eligible(validate_chunk(out, state=MEAN_STATE))[0]

    def test_infeasible_refused_not_repaired(self):
        rows = chunk(step=max_step_rad() * 6)
        out, rep = sanitize(rows, state=MEAN_STATE)
        assert rep.refused and out == rows

    def test_never_clips_a_joint(self):
        rows = chunk(); rows[2][11] = arm_joint_limit(11)[1] + 0.2
        out, _ = sanitize(rows, state=MEAN_STATE)
        assert max(r[11] for r in out) > arm_joint_limit(11)[1]
        assert has_fatal(validate_chunk(out))
