"""The execution path with fake i2rt robots: real Ruckig, no CAN, no model."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("ruckig")

from ..kuka.cameras import Frame                               # noqa: E402
from .conftest import MEAN_STATE, chunk                         # noqa: E402
from .contract import CAMERA_NAMES, JOINT_INDICES, arm_joint_limit  # noqa: E402
from .execute import (EXECUTE_REQUIRED, ExecSettings, ExecutionRun,  # noqa: E402
                      MonitorWorker, execution_blockers)
from .motion import MotionLimits, MotionOwner, MotionRefused, goal_problem  # noqa: E402


class FakeRobot:
    """Tracks the command perfectly, or not at all when `stuck`."""

    def __init__(self, q7):
        self.q = [float(v) for v in q7]
        self.commands = 0
        self.stuck = False
        self.closed = False

    def get_joint_pos(self):
        return list(self.q)

    def command_joint_pos(self, q):
        self.commands += 1
        if not self.stuck:
            self.q = [float(v) for v in q]

    def close(self):
        self.closed = True


def robots(state=MEAN_STATE):
    return {"left": FakeRobot(state[:7]), "right": FakeRobot(state[7:])}


FAST = MotionLimits(velocity=2.0, acceleration=6.0, jerk=60.0)


def config(**over):
    c = {"camera_mapping": {"top": 0, "left": 2, "right": 4},
         "gripper_limits_left": [1.0, -4.0], "gripper_limits_right": [1.0, -4.0],
         "rest_pose": MEAN_STATE, "estop_tested": "2026-09-28 by operator",
         "max_speed": 1.0, "max_acceleration": 3.0, "max_step_displacement": 0.5,
         "commanded_observed_tolerance_rad": 0.1, "observation_freshness_s": 0.5}
    c.update(over)
    return c


@pytest.fixture
def owner():
    o = MotionOwner(robots(), FAST, tolerance_rad=0.1).start()
    yield o
    o.stop()


class TestMotionOwner:
    def test_reaches_goal_smoothly(self, owner):
        goal = list(MEAN_STATE); goal[0] += 0.2; goal[12] -= 0.1
        owner.set_goal(goal)
        assert owner.wait_settled(5.0)
        assert owner.measured[0] == pytest.approx(goal[0], abs=1e-3)
        assert owner.measured[12] == pytest.approx(goal[12], abs=1e-3)
        assert owner.fault is None and owner.ticks > 5

    def test_goal_outside_limits_refused(self, owner):
        g = list(MEAN_STATE); g[1] = arm_joint_limit(1)[0] - 0.1
        with pytest.raises(MotionRefused):
            owner.set_goal(g)
        with pytest.raises(MotionRefused):
            owner.set_goal([0.0] * 7)

    def test_deviation_faults_and_brakes(self, owner):
        owner.robots["right"].stuck = True
        g = list(MEAN_STATE); g[9] += 0.6
        owner.set_goal(g)
        deadline = time.time() + 3
        while owner.fault is None and time.time() < deadline:
            time.sleep(0.02)
        assert owner.fault and "deviation" in owner.fault and owner.braking
        with pytest.raises(MotionRefused):
            owner.set_goal(MEAN_STATE)

    def test_no_park_after_fault_without_a_decision(self, owner):
        owner.fault = "test fault"
        with pytest.raises(MotionRefused):
            owner.park(MEAN_STATE, FAST)

    def test_brake_refuses_new_goals(self, owner):
        owner.brake()
        with pytest.raises(MotionRefused):
            owner.set_goal(MEAN_STATE)

    def test_park_moves_to_rest(self, owner):
        rest = list(MEAN_STATE); rest[3] += 0.1
        owner.brake()
        assert owner.park(rest, FAST, timeout_s=5.0)
        assert owner.measured[3] == pytest.approx(rest[3], abs=1e-3)

    def test_limits_refused_above_reference_ceiling(self):
        with pytest.raises(MotionRefused):
            MotionLimits(velocity=3.0)

    def test_goal_problem(self):
        assert goal_problem(MEAN_STATE) is None
        assert goal_problem(MEAN_STATE[:13])


class TestBlockers:
    def test_every_value_named(self):
        b = execution_blockers({})
        assert len(b) == len(EXECUTE_REQUIRED)

    def test_ranges(self):
        assert execution_blockers(config()) == []
        assert execution_blockers(config(max_speed=3.0))
        assert execution_blockers(config(commanded_observed_tolerance_rad=0))
        bad = list(MEAN_STATE); bad[1] = -2
        assert any("rest_pose" in x for x in execution_blockers(config(rest_pose=bad)))
        assert execution_blockers(config(start_pose=[0.0] * 3))

    def test_shipped_config_cannot_arm(self):
        from .experiment import flatten_config, load_config
        assert execution_blockers(flatten_config(load_config(local=False)))


def frames():
    now = time.monotonic()
    return {c: Frame(c, b"\xff\xd8x", now, time.time(), 1, 640, 360) for c in CAMERA_NAMES}


class Policy:
    """Proposes a slow drift of every arm joint from the state it is shown."""

    def __init__(self, step=0.004, fail=False):
        self.step, self.fail, self.seen = step, fail, []

    def __call__(self, obs):
        self.seen.append(obs)
        if self.fail:
            return {"ok": False, "error": "server down"}
        return {"ok": True, "rows": chunk(obs["state"], step=self.step)}


def run_for(owner, policy, *, cycles=3, monitor=None, **settings):
    s = ExecSettings(stop_file="/nonexistent/yam_stop", max_seconds=30, **settings)
    r = ExecutionRun(owner, propose=policy, grab_frames=frames, settings=s,
                     config=config(), task="stack the blocks", monitor=monitor)
    for i in range(cycles):
        r.cycle(i + 1)
    return r


class TestExecution:
    def test_refuses_to_construct_unarmed(self, owner):
        with pytest.raises(MotionRefused):
            ExecutionRun(owner, propose=Policy(), grab_frames=frames,
                         settings=ExecSettings(), config={}, task="t")

    def test_executes_bounded_prefixes(self, owner):
        start = list(owner.measured)
        r = run_for(owner, Policy(), cycles=3, steps_per_chunk=4)
        assert [x.outcome for x in r.records] == ["executed"] * 3
        assert all(x.steps_executed == 4 for x in r.records)
        owner.wait_settled(3.0)
        moved = owner.measured[0] - start[0]
        # each chunk starts from the MEASURED pose, which trails the reference,
        # so the arm travels a little less than the sum of the proposals
        assert 0.5 * 3 * 4 * 0.004 < moved <= 3 * 4 * 0.004 + 1e-3

    def test_policy_sees_commanded_gripper(self, owner):
        owner.robots["left"].q[6] = 0.1           # encoder says nearly closed
        p = Policy()
        run_for(owner, p, cycles=1)
        assert p.seen[0]["state"][6] == pytest.approx(MEAN_STATE[6], abs=1e-6)
        assert set(p.seen[0]["images"]) == set(CAMERA_NAMES)

    def test_server_error_holds(self, owner):
        r = run_for(owner, Policy(fail=True), cycles=2)
        assert all(x.outcome == "hold" and x.steps_executed == 0 for x in r.records)

    def test_fast_proposal_is_time_scaled_not_rejected(self, owner):
        def fast(obs):                  # 0.1 rad/step for 4 rows, then still
            rows = chunk(obs["state"], step=0.1, n=4)
            return {"ok": True, "rows": rows + [list(rows[-1])] * 12}
        r = run_for(owner, fast, cycles=1, steps_per_chunk=4)
        rec = r.records[0]
        assert rec.outcome == "executed" and rec.sanitizer_changed

    def test_large_prefix_blocked_by_displacement_cap(self, owner):
        s = ExecSettings(stop_file="/nonexistent/x", steps_per_chunk=16)
        r = ExecutionRun(owner, propose=Policy(step=0.05), grab_frames=frames,
                         settings=s, config=config(max_step_displacement=0.1),
                         task="t")
        rec = r.cycle(1)
        assert rec.outcome == "hold" and "max_step_displacement" in rec.reason

    def test_fatal_proposal_holds(self, owner):
        def bad(obs):
            rows = chunk(obs["state"]); rows[3][1] = -3.0
            return {"ok": True, "rows": rows}
        r = run_for(owner, bad, cycles=1)
        assert r.records[0].outcome == "hold"

    def test_stop_file_stops(self, owner, tmp_path):
        stop = tmp_path / "stop"
        s = ExecSettings(stop_file=str(stop), max_seconds=30)
        r = ExecutionRun(owner, propose=Policy(), grab_frames=frames, settings=s,
                         config=config(), task="t")
        stop.touch()
        out = r.run()
        assert out["cycles"] == 0 and "stop file" in out["stop_reason"]

    def test_fault_stops_the_run(self, owner):
        owner.fault = "deviation"
        r = ExecutionRun(owner, propose=Policy(), grab_frames=frames,
                         settings=ExecSettings(stop_file="/nonexistent/x"),
                         config=config(), task="t")
        assert "motion fault" in r.run()["stop_reason"]


class FakeOutcome:
    def __init__(self, steps, astra=None, called=False, controller="pi05"):
        self.executed_steps, self.astra_decision = steps, astra
        self.astra_called, self.controller = called, controller


class TestMonitorGate:
    def worker(self, outcome, at=None):
        w = MonitorWorker(pipeline=None)
        w.latest, w.latest_at = outcome, at if at is not None else time.monotonic()
        w.post = lambda snap: None
        return w

    def test_shadow_keeps_fixed_prefix(self, owner):
        r = run_for(owner, Policy(), cycles=1, steps_per_chunk=5,
                    monitor=self.worker(FakeOutcome(1)))
        assert r.records[0].steps_executed == 5

    def test_gating_uses_monitor_steps(self, owner):
        r = run_for(owner, Policy(), cycles=1, steps_per_chunk=8, monitor_gate=True,
                    monitor=self.worker(FakeOutcome(2)))
        assert r.records[0].steps_executed == 2

    def test_stale_decision_holds(self, owner):
        r = run_for(owner, Policy(), cycles=1, monitor_gate=True, monitor_max_age_s=1.0,
                    monitor=self.worker(FakeOutcome(8), at=time.monotonic() - 5))
        assert r.records[0].outcome == "hold"

    def test_no_decision_yet_holds(self, owner):
        r = run_for(owner, Policy(), cycles=1, monitor_gate=True,
                    monitor=self.worker(None))
        assert r.records[0].outcome == "hold"

    def test_astra_stop_ends_the_run(self, owner):
        w = self.worker(FakeOutcome(0, {"ok": True, "decision": {"mode": "stop"}}, True))
        s = ExecSettings(stop_file="/nonexistent/x")
        r = ExecutionRun(owner, propose=Policy(), grab_frames=frames, settings=s,
                         config=config(), task="t", monitor=w)
        out = r.run()
        assert out["stop_reason"] == "Astra requested stop"

    def test_worker_runs_the_real_pipeline(self):
        from ..kuka.vlm_backends import MockMonitorBackend
        from .pipeline import PolicyMode, YamMonitorSchedule, YamPolicyPipeline
        rd = {"phase": "approach", "progress": "normal", "target_visible": True,
              "grasp_confirmed": False, "slip_detected": False, "intent": "aligned",
              "confidence": 0.9, "execute_steps": 3, "escalate": False,
              "evidence": "ok"}
        pipe = YamPolicyPipeline(PolicyMode.PI05_LOCAL_MONITOR,
                                 backend=MockMonitorBackend([rd]), shadow=False,
                                 schedule=YamMonitorSchedule(min_interval_s=1e-6,
                                                             max_interval_s=1e-6,
                                                             measured_latency_s=1e-6))
        w = MonitorWorker(pipe).start()
        w.post(dict(state=MEAN_STATE, proposed_steps=16, proposed_chunk=chunk()))
        deadline = time.time() + 3
        while w.latest is None and time.time() < deadline:
            time.sleep(0.01)
        w.stop()
        assert w.latest.executed_steps == 3


def test_every_joint_moves_through_the_owner_only(owner):
    before = {a: r.commands for a, r in owner.robots.items()}
    time.sleep(0.1)
    assert all(owner.robots[a].commands > before[a] for a in before)
    assert JOINT_INDICES
