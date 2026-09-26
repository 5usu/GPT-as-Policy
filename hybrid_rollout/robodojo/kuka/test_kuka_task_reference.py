"""A task reference must anchor the intent gate without becoming a template."""
import json

import pytest

from .packet import build_packet
from .task_reference import (GRIP_THRESHOLD, ReferenceRefused, TaskReference,
                             from_episode, load, segment_phases)

# approach open -> close -> transport holding -> release -> retreat
ROWS = ([[0, 0, 0, 0, 0, 0, 0.0]] * 10
        + [[5, -3, 2, 0, 0, 0, 0.0]] * 10
        + [[5, -3, 2, 0, 0, 0, 1.0]] * 8
        + [[20, -10, 6, 0, 0, 0, 1.0]] * 12
        + [[20, -10, 6, 0, 0, 0, 0.0]] * 6)


def ref(**kw):
    base = dict(task="open the dishwasher door", episode_id="ep0498",
                rows=ROWS, verified_by="BillyChern", verified_successful=True)
    base.update(kw)
    return from_episode(**base)


class TestVerificationIsMandatory:
    """A reference asserts 'this succeeded'. Nobody may assert that silently."""

    def test_unverified_episode_is_refused(self):
        with pytest.raises(ReferenceRefused) as e:
            ref(verified_successful=False)
        assert "not marked successful" in str(e.value)

    def test_a_verifier_must_be_named(self):
        with pytest.raises(ReferenceRefused) as e:
            ref(verified_by="  ")
        assert "no named verifier" in str(e.value)

    def test_the_task_must_be_stated(self):
        with pytest.raises(ReferenceRefused):
            ref(task="")

    def test_the_verifier_appears_in_the_prompt(self):
        assert "verified by BillyChern" in ref().render()


class TestItCannotBecomeATemplate:
    """The failure mode this guards: the reviewer copying the reference, or
    treating any divergence from it as failure."""

    def test_raw_joint_rows_are_never_rendered(self):
        text = ref().render()
        for token in ("t+00", "t+01", "[20", "-10.0"):
            assert token not in text, \
                "per-step targets must not reach the prompt; they would be " \
                "copied as an 'edit' from the wrong starting pose"

    def test_it_says_outright_that_this_is_not_the_current_scene(self):
        text = ref().render()
        assert "NOT the current scene" in text
        assert "NOT a trajectory to reproduce" in text

    def test_divergence_is_explicitly_not_a_takeover_reason(self):
        text = ref().render()
        assert "Divergence from this reference is NOT evidence of failure" in text
        assert "NOT a takeover reason" in text

    def test_it_explains_why_the_values_are_withheld(self):
        assert "different starting pose" in ref().render()

    def test_the_audit_records_what_was_withheld(self):
        d = ref().to_log()
        assert "raw joint rows" in d["withheld_from_prompt"]


class TestPhaseSegmentation:
    """Gripper transitions are the only unambiguous task landmark available
    without a scene model."""

    def test_phases_split_at_gripper_transitions(self):
        ph = segment_phases(ROWS)
        assert [p.name for p in ph] == [
            "approach (open)", "holding (closed)", "retreat (released)"]

    def test_a_release_is_not_mislabelled_as_another_approach(self):
        """Regression: the released flag was set after naming, so the retreat
        phase read as a second approach."""
        assert segment_phases(ROWS)[-1].name == "retreat (released)"

    def test_net_joint_travel_is_reported_per_phase(self):
        hold = segment_phases(ROWS)[1]
        assert hold.net_joint_deg[0] == pytest.approx(15.0)
        assert hold.net_joint_deg[1] == pytest.approx(-7.0)

    def test_duration_comes_from_the_control_rate(self):
        ph = segment_phases(ROWS)
        assert ph[0].n_steps == 20 and ph[0].duration_s == pytest.approx(20 / 30)

    def test_an_episode_with_no_gripper_column_still_segments(self):
        ph = segment_phases([[0] * 6, [1] * 6, [2] * 6])
        assert len(ph) == 1 and ph[0].name == "segment 1"

    def test_an_empty_episode_yields_no_phases(self):
        assert segment_phases([]) == []

    def test_the_threshold_is_the_documented_one(self):
        assert GRIP_THRESHOLD == 0.5


class TestPacketIntegration:

    def _packet(self, **kw):
        return build_packet(task_instruction="open the dishwasher door",
                            observation_id="live:1", state=[0.0] * 7,
                            chunk=[[0.0] * 7] * 50,
                            provenance="model_predicted", frames={}, **kw)

    def test_absent_by_default_so_upstream_stays_comparable(self):
        """Upstream gives its reviewer no reference; a run WITH one is not
        comparable to the published 48% / 62.60 hybrid numbers."""
        p = self._packet()
        assert p["reference"] is None
        assert "TASK REFERENCE" not in p["user_text"]

    def test_the_reference_precedes_the_proposal(self):
        """It must read as what the task is, not as a comment on this chunk."""
        t = self._packet(reference=ref())["user_text"]
        assert t.index("TASK REFERENCE") < t.index("ABSOLUTE joint targets")

    def test_provenance_still_says_the_proposal_was_not_executed(self):
        """Adding a successful reference must not blur the line between what
        ran and what is only proposed."""
        t = self._packet(reference=ref())["user_text"]
        assert "has NOT been executed" in t
        assert "nothing you can see is a consequence of it" in t

    def test_the_reference_is_recorded_in_the_audit(self):
        p = self._packet(reference=ref())
        assert p["reference"]["episode_id"] == "ep0498"
        assert p["reference"]["verified_by"] == "BillyChern"
        assert p["reference"]["schema"] == "kuka.task_reference.v1"

    def test_subgoals_reach_the_prompt_in_order(self):
        r = ref(subgoals=["reach the handle", "close on it", "swing it open"])
        t = self._packet(reference=r)["user_text"]
        assert t.index("reach the handle") < t.index("swing it open")


class TestLoadingFromDisk:

    def test_a_verified_file_loads(self, tmp_path):
        f = tmp_path / "r.json"
        f.write_text(json.dumps({
            "task": "open the dishwasher door", "episode_id": "ep0498",
            "verified_successful": True, "verified_by": "BillyChern",
            "subgoals": ["reach"], "rows": ROWS}))
        r = load(str(f))
        assert r.episode_id == "ep0498" and r.phases

    def test_an_unverified_file_is_refused(self, tmp_path):
        f = tmp_path / "r.json"
        f.write_text(json.dumps({
            "task": "t", "episode_id": "e", "verified_by": "b", "rows": ROWS}))
        with pytest.raises(ReferenceRefused):
            load(str(f))

    def test_a_file_missing_fields_names_them(self, tmp_path):
        f = tmp_path / "r.json"
        f.write_text(json.dumps({"task": "t", "rows": ROWS}))
        with pytest.raises(ReferenceRefused) as e:
            load(str(f))
        assert "episode_id" in str(e.value) and "verified_by" in str(e.value)


class TestItReachesAstraOnEscalation:
    """Wiring it into build_packet is useless if escalation never passes it."""

    def test_an_escalation_forwards_the_reference_to_astra(self):
        from .pipeline import PolicyPipeline
        from .vlm_monitor import MonitorPolicy
        from .test_kuka_vlm import MockMonitorBackend, fast_schedule, reading

        seen = {}

        def fake_astra(pkt):
            seen.update(pkt)
            return {"ok": True, "decision": {"mode": "student", "steps": 3,
                                             "request_id": pkt["request_id"]}}

        seq = [reading()] * 2 + [reading(progress="failed")] * 3
        p = PolicyPipeline(
            "pi05_local_monitor_astra", schedule=fast_schedule(),
            backend=MockMonitorBackend(seq), shadow=False,
            policy=MonitorPolicy(escalate_after_adverse=3,
                                 max_reading_age_s=60.0),
            astra_review=fake_astra, task_reference=ref())
        t = 1000.0
        for i in range(len(seq)):
            p.step(state=[0.0] * 7, proposed_steps=5,
                   proposed_chunk=[[0.0] * 7] * 50,
                   frames=["f"], frames_meta={"base": {}},
                   task="open the dishwasher door", now=t + i)
        assert seen, "Astra was never called"
        assert "TASK REFERENCE" in seen["user_text"]
        assert seen["reference"]["episode_id"] == "ep0498"

    def test_metrics_record_whether_a_reference_was_in_play(self):
        from .pipeline import PolicyPipeline
        assert PolicyPipeline("pi05_only").metrics()["task_reference"] is None
        p = PolicyPipeline("pi05_only", task_reference=ref())
        assert p.metrics()["task_reference"]["episode_id"] == "ep0498"
