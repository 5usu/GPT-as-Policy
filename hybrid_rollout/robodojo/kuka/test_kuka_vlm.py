"""Local-VLM monitor gate: schema, clamping, hysteresis, fail-safe, routing.

Mock backends only. No model service, no weights, no paid calls, no hardware.
"""
from __future__ import annotations

import time

import pytest

from .pipeline import (COMPAT_ALIASES, CycleOutcome, Event, EventDetector,
                       MonitorSchedule, PolicyMode, PolicyPipeline, resolve_mode)
from .vlm_backends import (DEFAULT_MODEL, EDGE_MODEL, MockMonitorBackend,
                           OpenAICompatibleBackend, UnconfiguredBackend,
                           VlmConfig, make_backend)
from .vlm_monitor import (Disposition, MonitorGate, MonitorPolicy, MonitorRejected,
                          parse_reading, response_schema)

STATE = [-76.55, -94.75, 66.60, 8.53, 19.30, 5.19, 0.0]


def reading(**over):
    r = {"phase": "approach", "progress": "normal", "target_visible": True,
         "grasp_confirmed": False, "slip_detected": False, "intent": "aligned",
         "confidence": 0.9, "execute_steps": 5, "escalate": False,
         "evidence": "handle visible, gripper approaching"}
    r.update(over)
    return r


class TestSchemaValidation:
    def test_valid_reading_parses(self):
        r = parse_reading(reading())
        assert r.progress.value == "normal" and r.execute_steps_raw == 5

    def test_json_string_parses(self):
        import json
        assert parse_reading(json.dumps(reading())).confidence == 0.9

    def test_malformed_json_rejected(self):
        with pytest.raises(MonitorRejected) as e:
            parse_reading("{not json")
        assert e.value.code == "malformed_json"

    @pytest.mark.parametrize("field", ["phase", "progress", "intent",
                                       "confidence", "execute_steps", "evidence"])
    def test_missing_field_rejected(self, field):
        r = reading(); del r[field]
        with pytest.raises(MonitorRejected) as e:
            parse_reading(r)
        assert e.value.code == "missing_fields"

    def test_unexpected_field_rejected(self):
        """Extra keys are refused, not ignored -- a model inventing a field is a
        signal that it is not answering the contract."""
        with pytest.raises(MonitorRejected) as e:
            parse_reading(reading(surprise="hello"))
        assert e.value.code == "unexpected_fields"

    def test_bad_enum_rejected(self):
        with pytest.raises(MonitorRejected) as e:
            parse_reading(reading(progress="vibes"))
        assert e.value.code == "bad_enum"

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "high", True])
    def test_confidence_out_of_range_rejected(self, bad):
        with pytest.raises(MonitorRejected):
            parse_reading(reading(confidence=bad))

    @pytest.mark.parametrize("bad", [-1, 2.5, "five", True])
    def test_bad_execute_steps_rejected(self, bad):
        with pytest.raises(MonitorRejected):
            parse_reading(reading(execute_steps=bad))

    def test_empty_evidence_rejected(self):
        with pytest.raises(MonitorRejected) as e:
            parse_reading(reading(evidence="   "))
        assert e.value.code == "empty_evidence"

    def test_non_boolean_flag_rejected(self):
        with pytest.raises(MonitorRejected):
            parse_reading(reading(slip_detected="yes"))

    def test_schema_is_strict(self):
        assert response_schema()["additionalProperties"] is False


class TestClampingIsIndependentOfTheModel:
    def test_absurd_request_is_clamped(self):
        g = MonitorGate()
        d = g.evaluate(parse_reading(reading(execute_steps=999)),
                       proposed_steps=50)
        assert d.execute_steps == g.policy.max_steps_normal
        assert d.clamped_from == 999

    def test_high_confidence_does_not_widen_the_bound(self):
        g = MonitorGate()
        a = g.evaluate(parse_reading(reading(execute_steps=99, confidence=0.99)),
                       proposed_steps=50).execute_steps
        b = MonitorGate().evaluate(
            parse_reading(reading(execute_steps=99, confidence=0.01)),
            proposed_steps=50).execute_steps
        assert a == b, "confidence is advisory and must not relax a limit"

    def test_proposed_steps_also_bounds(self):
        d = MonitorGate().evaluate(parse_reading(reading(execute_steps=99)),
                                   proposed_steps=2)
        assert d.execute_steps == 2

    def test_contact_phase_gets_a_shorter_leash(self):
        g = MonitorGate()
        d = g.evaluate(parse_reading(reading(phase="grasp", execute_steps=99)),
                       proposed_steps=50)
        assert d.execute_steps == g.policy.max_steps_contact

    def test_uncertain_reading_is_shortened(self):
        g = MonitorGate()
        d = g.evaluate(parse_reading(reading(progress="uncertain",
                                             execute_steps=99)),
                       proposed_steps=50)
        assert d.execute_steps <= g.policy.max_steps_uncertain
        assert d.disposition is Disposition.SHORTEN

    def test_never_negative(self):
        d = MonitorGate().evaluate(parse_reading(reading(execute_steps=0)),
                                   proposed_steps=50)
        assert d.execute_steps == 0


class TestHysteresisPreventsTakeoverOnOneFrame:
    def test_single_adverse_frame_shortens_but_does_not_escalate(self):
        g = MonitorGate()
        d = g.evaluate(parse_reading(reading(progress="failed")),
                       proposed_steps=10)
        assert d.disposition is Disposition.SHORTEN
        assert d.escalate is False

    def test_escalation_requires_persistence(self):
        g = MonitorGate(MonitorPolicy(escalate_after_adverse=3))
        out = [g.evaluate(parse_reading(reading(progress="failed")),
                          proposed_steps=10).disposition for _ in range(3)]
        assert out[0] is Disposition.SHORTEN
        assert out[1] is Disposition.SHORTEN
        assert out[2] is Disposition.ESCALATE

    def test_a_good_frame_resets_the_streak(self):
        g = MonitorGate(MonitorPolicy(escalate_after_adverse=3))
        g.evaluate(parse_reading(reading(progress="failed")), proposed_steps=10)
        g.evaluate(parse_reading(reading(progress="failed")), proposed_steps=10)
        g.evaluate(parse_reading(reading()), proposed_steps=10)
        assert g.adverse_streak == 0
        d = g.evaluate(parse_reading(reading(progress="failed")), proposed_steps=10)
        assert d.disposition is Disposition.SHORTEN, "streak restarted"

    def test_persistent_uncertainty_holds_rather_than_escalating(self):
        g = MonitorGate(MonitorPolicy(hold_after_uncertain=3))
        for _ in range(2):
            g.evaluate(parse_reading(reading(progress="uncertain")),
                       proposed_steps=10)
        d = g.evaluate(parse_reading(reading(progress="uncertain")),
                       proposed_steps=10)
        assert d.disposition is Disposition.HOLD_REOBSERVE
        assert d.execute_steps == 0 and d.escalate is False

    def test_slip_counts_as_adverse(self):
        g = MonitorGate(MonitorPolicy(escalate_after_adverse=2))
        g.evaluate(parse_reading(reading(slip_detected=True)), proposed_steps=10)
        d = g.evaluate(parse_reading(reading(slip_detected=True)), proposed_steps=10)
        assert d.disposition is Disposition.ESCALATE

    def test_hand_back_requires_verified_recovery(self):
        g = MonitorGate(MonitorPolicy(escalate_after_adverse=1,
                                      hand_back_after_normal=2))
        g.evaluate(parse_reading(reading(progress="failed")), proposed_steps=10)
        assert g.escalated and g.may_hand_back()[0] is False
        g.evaluate(parse_reading(reading()), proposed_steps=10)
        assert g.may_hand_back()[0] is False, "one good frame is not recovery"
        g.evaluate(parse_reading(reading()), proposed_steps=10)
        assert g.may_hand_back()[0] is True
        g.hand_back()
        assert g.escalated is False


class TestFailSafe:
    def test_no_reading_executes_nothing(self):
        d = MonitorGate().evaluate(None, proposed_steps=10, error="service down")
        assert d.disposition is Disposition.FAIL_SAFE and d.execute_steps == 0

    def test_stale_reading_executes_nothing(self):
        g = MonitorGate(MonitorPolicy(max_reading_age_s=0.5))
        r = parse_reading(reading())
        r.observed_at = time.time() - 5.0
        d = g.evaluate(r, proposed_steps=10)
        assert d.disposition is Disposition.FAIL_SAFE
        assert "already left" in d.reason

    def test_fail_safe_is_a_hold_not_a_fault(self):
        """Unreachable monitor -> stop moving. It must not latch the arm."""
        d = MonitorGate().fail_safe("timeout")
        assert d.execute_steps == 0
        assert d.disposition is Disposition.FAIL_SAFE

    def test_malformed_response_reaches_fail_safe_through_the_pipeline(self):
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend([{"garbage": True}]),
                           shadow=False)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps == 0 and out.monitor_error

    def test_backend_exception_reaches_fail_safe(self):
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend([], fail_with="conn refused"),
                           shadow=False)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps == 0 and "conn refused" in out.monitor_error


class TestModeRouting:
    def test_three_modes_exist(self):
        assert [m.value for m in PolicyMode] == [
            "pi05_only", "pi05_local_monitor", "pi05_local_monitor_astra"]

    @pytest.mark.parametrize("alias,expected", list(COMPAT_ALIASES.items()))
    def test_compat_aliases_resolve(self, alias, expected):
        assert resolve_mode(alias) is expected

    def test_unknown_mode_has_no_fallback(self):
        with pytest.raises(ValueError):
            resolve_mode("something_else")

    def test_pi05_only_never_calls_the_monitor(self):
        backend = MockMonitorBackend([reading()])
        p = PolicyPipeline("pi05_only", backend=backend)
        for _ in range(3):
            p.step(state=STATE, proposed_steps=10)
        assert backend._i == 0, "pi05_only must not consult the monitor"
        assert all(r.gate is None for r in p.records)

    def test_pi05_only_semantics_unchanged(self):
        p = PolicyPipeline("pi05_only", default_steps=5)
        out = p.step(state=STATE, proposed_steps=50)
        assert out.executed_steps == out.baseline_steps == 5

    def test_local_monitor_never_calls_astra(self):
        calls = []
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend(
                               [reading(progress="failed")] * 6),
                           astra_review=lambda pkt: calls.append(pkt),
                           shadow=False)
        for _ in range(6):
            p.step(state=STATE, proposed_steps=10,
                   now=time.time() + len(p.records))
        assert calls == [], "the middle mode must never escalate to Astra"

    def test_astra_mode_escalates_after_persistence(self):
        calls = []
        # max_reading_age_s is widened ONLY because this test synthesises a
        # timeline: the mock stamps each reading with the real clock while `now`
        # is fabricated, so the default 1.0s freshness bound would (correctly)
        # call the later readings stale. See
        # test_fabricated_future_now_trips_the_freshness_check.
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend(
                               [reading(progress="failed")] * 4),
                           policy=MonitorPolicy(escalate_after_adverse=3,
                                                max_reading_age_s=60.0),
                           astra_review=lambda pkt: (calls.append(pkt),
                                                     {"ok": True,
                                                      "mode": "student"})[1],
                           shadow=False)
        t = time.time()
        for i in range(3):
            p.step(state=STATE, proposed_steps=10, now=t + i)
        assert len(calls) == 1, "exactly one escalation after 3 adverse"
        assert p.controller == "astra"

    def test_fabricated_future_now_trips_the_freshness_check(self):
        """A reading that is old relative to `now` fails safe, even mid-streak.

        This is the behaviour that made the escalation test above need an
        explicit freshness override: staleness outranks an accumulating adverse
        streak, because a status describing a scene the arm has already left
        must not be used to justify a takeover.
        """
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend(
                               [reading(progress="failed")] * 3),
                           policy=MonitorPolicy(escalate_after_adverse=2,
                                                max_reading_age_s=1.0),
                           astra_review=lambda pkt: {"ok": True},
                           shadow=False)
        t = time.time()
        p.step(state=STATE, proposed_steps=10, now=t)
        out = p.step(state=STATE, proposed_steps=10, now=t + 30)
        assert out.gate["disposition"] == "fail_safe"
        assert out.executed_steps == 0
        assert out.escalated is False, "staleness must not escalate"

    def test_preflight_blocks_when_no_monitor_service(self):
        p = PolicyPipeline("pi05_local_monitor")
        ok, blockers = p.preflight()
        assert ok is False and "unavailable" in blockers[0]

    def test_astra_mode_needs_a_reviewer(self):
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend([reading()]))
        ok, blockers = p.preflight()
        assert any("Astra" in b for b in blockers)


class TestShadowIsDefault:
    def test_shadow_on_by_default(self):
        assert PolicyPipeline("pi05_local_monitor").shadow is True
        assert MonitorGate().shadow is True

    def test_shadow_does_not_alter_the_executed_prefix(self):
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend(
                               [reading(progress="failed", execute_steps=0)]),
                           default_steps=5, shadow=True)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps == out.baseline_steps == 5
        assert out.gate["execute_steps"] < 5, "it WOULD have shortened"
        assert "SHADOW" in out.reason

    def test_gating_requires_explicit_switch(self):
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend(
                               [reading(progress="failed", execute_steps=0)]),
                           default_steps=5, shadow=False)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps != out.baseline_steps


class TestSchedulingStaysOutOfTheControlLoop:
    def test_rate_is_2_to_5_hz(self):
        s = MonitorSchedule()
        assert 2.0 <= 1.0 / s.max_interval_s and 1.0 / s.min_interval_s <= 5.0

    def test_not_evaluated_every_tick(self):
        d = EventDetector(MonitorSchedule(min_interval_s=0.2, max_interval_s=0.5))
        t = time.time()
        due = [d.due(state=STATE, now=t + i * 0.004)[0] for i in range(100)]
        assert sum(due) <= 2, "250 Hz ticks must not each trigger inference"

    def test_gripper_close_is_a_semantic_event(self):
        d = EventDetector()
        t = time.time()
        d.due(state=STATE, now=t)
        closed = list(STATE[:6]) + [1.0]
        ok, ev = d.due(state=closed, now=t + 0.3)
        assert ok and ev is Event.GRIPPER_CLOSE

    def test_stall_is_detected(self):
        d = EventDetector(MonitorSchedule(stall_cycles=3, min_interval_s=0.0))
        t = time.time()
        ev = None
        for i in range(6):
            _ok, e = d.due(state=STATE, now=t + i * 0.01)
            ev = e or ev
        assert ev in (Event.STALLED_MOTION, Event.PERIODIC)


class TestBackends:
    def test_model_choices_recorded(self):
        assert DEFAULT_MODEL == "Qwen/Qwen3-VL-2B-Instruct"
        assert EDGE_MODEL == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def test_default_backend_refuses(self):
        b = UnconfiguredBackend()
        assert b.probe().available is False
        with pytest.raises(MonitorRejected):
            b.observe()

    def test_edge_config_swaps_model_without_controller_change(self):
        a = make_backend("a800")
        j = make_backend("jetson")
        assert a.model == DEFAULT_MODEL and j.model == EDGE_MODEL
        assert type(a) is type(j), "same client, different configuration"

    def test_probe_detects_a_missing_service(self):
        def boom(url, body, timeout):
            raise OSError("connection refused")
        b = OpenAICompatibleBackend(VlmConfig(), transport=boom)
        p = b.probe()
        assert p.available is False and "will not gate motion" in p.reason

    def test_request_carries_multiple_frames_oldest_first(self):
        b = OpenAICompatibleBackend(VlmConfig())
        body = b.build_body(frames=["data:a", "data:b"], state_text="s",
                            intent_text="i")
        texts = [c["text"] for c in body["messages"][1]["content"]
                 if c["type"] == "text"]
        assert any("oldest" in t for t in texts)
        assert any("current" in t for t in texts)

    def test_request_pins_the_strict_schema(self):
        b = OpenAICompatibleBackend(VlmConfig())
        body = b.build_body(frames=[], state_text="", intent_text="")
        assert body["response_format"]["json_schema"]["strict"] is True

    def test_credential_never_in_the_body(self, monkeypatch):
        monkeypatch.setenv("LOCAL_VLM_API_KEY", "super-secret")
        import json
        b = OpenAICompatibleBackend(VlmConfig())
        assert "super-secret" not in json.dumps(
            b.build_body(frames=[], state_text="", intent_text=""))

    def test_mock_is_always_flagged(self):
        m = MockMonitorBackend([reading()])
        assert m.is_mock is True
        assert m.observe(frames=[], state_text="", intent_text="",
                         timeout_s=1).backend == "mock"


class TestMetricsAreAuditable:
    def test_records_every_required_field(self):
        rows = []
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend([reading()]),
                           astra_review=lambda pkt: {"ok": True},
                           on_record=rows.append)
        p.step(state=STATE, proposed_steps=10, episode_id="ep0", task="open door")
        row = rows[0]
        for k in ("mode", "cycle", "event", "proposed_steps", "executed_steps",
                  "baseline_steps", "gate", "monitor_error", "escalated",
                  "astra_called", "handed_back", "controller", "shadow",
                  "episode_id", "task", "monitor_latency_s"):
            assert k in row, k

    def test_metrics_report_clamping_and_streaks(self):
        p = PolicyPipeline("pi05_local_monitor",
                           backend=MockMonitorBackend(
                               [reading(execute_steps=99)] * 2), shadow=False)
        t = time.time()
        for i in range(2):
            p.step(state=STATE, proposed_steps=50, now=t + i)
        m = p.metrics()
        assert m["gate"]["clamped_decisions"] >= 1
        assert "uncertain_streak" in m["gate"]
        assert m["compat_aliases"]

    def test_records_are_append_only(self):
        p = PolicyPipeline("pi05_only")
        for _ in range(3):
            p.step(state=STATE, proposed_steps=5)
        assert len(p.records) == 3
        assert [r.cycle for r in p.records] == [1, 2, 3]


class TestExistingSafetyPathUntouched:
    """Regression: the VLM layer must not have altered the 250 Hz path."""

    def test_rsi_gateway_has_no_vlm_import(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "rsi_gateway.py").read_text()
        assert "vlm_" not in src and "pipeline" not in src

    def test_execution_state_machine_has_no_vlm_import(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "execution.py").read_text()
        assert "vlm_" not in src and "PolicyPipeline" not in src

    def test_otg_has_no_vlm_import(self):
        import pathlib
        src = (pathlib.Path(__file__).parent / "otg.py").read_text()
        assert "vlm_" not in src

    def test_execution_modes_unchanged(self):
        from .safety import COMMAND_CAPABLE_MODES, Mode
        assert [m.value for m in Mode] == [
            "replay", "live_shadow", "reviewed_execution", "astra_direct"]
        assert COMMAND_CAPABLE_MODES == frozenset(
            {Mode.REVIEWED_EXECUTION, Mode.ASTRA_DIRECT})

    def test_emittable_decision_modes_unchanged(self):
        from .loop import EXECUTABLE_DECISION_MODES
        assert EXECUTABLE_DECISION_MODES == frozenset(
            {"student", "astra_direct_joint"})

    def test_monitor_latency_cannot_delay_a_cycle(self):
        """The gate reads a cached decision; it never awaits inference."""
        slow = MockMonitorBackend([reading()], latency_s=5.0)
        p = PolicyPipeline("pi05_local_monitor", backend=slow, shadow=True)
        t0 = time.monotonic()
        p.step(state=STATE, proposed_steps=10)
        assert time.monotonic() - t0 < 1.0, "mock must not actually sleep"
        assert p.records[0].monitor_latency_s == 5.0, "latency is RECORDED"


class TestOfflineIntegration:
    """Recorded episode -> chunk -> monitor -> (escalate) -> Astra, all mocked."""

    def _episode(self):
        import json
        import pathlib
        fix = pathlib.Path(__file__).parent / "fixtures"
        chunks = json.loads((fix / "chunks.json").read_text())
        rows = next(v for k, v in chunks.items() if not k.startswith("_"))
        return rows

    def test_normal_run_stays_with_pi05(self):
        rows = self._episode()
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend([reading()] * 8),
                           policy=MonitorPolicy(max_reading_age_s=60.0),
                           astra_review=lambda pkt: {"ok": True}, shadow=False)
        t = time.time()
        for i in range(6):
            p.step(state=rows[i][:7], proposed_steps=len(rows), now=t + i,
                   episode_id="ep000000", task="open the white dishwasher")
        m = p.metrics()
        assert m["astra_calls"] == 0
        assert m["cycles_by_controller"].get("pi05") == 6
        assert m["steps_executed"] > 0

    def test_degrading_run_escalates_then_hands_back(self):
        rows = self._episode()
        seq = ([reading()] * 2 + [reading(progress="failed")] * 3
               + [reading()] * 3)
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend(seq),
                           policy=MonitorPolicy(escalate_after_adverse=3,
                                                hand_back_after_normal=2,
                                                max_reading_age_s=60.0),
                           astra_review=lambda pkt: {"ok": True,
                                                     "mode": "student",
                                                     "steps": 2},
                           shadow=False)
        t = time.time()
        for i in range(len(seq)):
            p.step(state=rows[min(i, len(rows) - 1)][:7],
                   proposed_steps=len(rows), now=t + i, episode_id="ep000000")
        m = p.metrics()
        assert m["escalations"] == 1 and m["astra_calls"] == 1
        assert m["hand_backs"] == 1, "control returns to pi0.5 after recovery"
        assert p.controller == "pi05"

    def test_shadow_run_changes_nothing_but_records_everything(self):
        rows = self._episode()
        seq = [reading(progress="failed", execute_steps=0)] * 4
        p = PolicyPipeline("pi05_local_monitor_astra",
                           backend=MockMonitorBackend(seq),
                           policy=MonitorPolicy(escalate_after_adverse=2,
                                                max_reading_age_s=60.0),
                           astra_review=lambda pkt: {"ok": True},
                           default_steps=5, shadow=True)
        t = time.time()
        for i in range(4):
            p.step(state=rows[i][:7], proposed_steps=50, now=t + i)
        assert all(r.executed_steps == r.baseline_steps for r in p.records)
        assert p.metrics()["gate"]["dispositions"].get("escalate", 0) >= 1, \
            "the gate still RECORDS what it would have done"
