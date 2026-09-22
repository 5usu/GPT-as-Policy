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


def fast_schedule():
    """Evaluate on every synthetic tick.

    The shipped default is paced from MEASURED device latency (~6 s for a 2B on
    an AGX Orin), so a test advancing `now` by one second would never come due.
    These tests exercise gate logic, not scheduling; TestSchedulingStaysOutOfThe
    ControlLoop covers the real pacing.
    """
    return MonitorSchedule(min_interval_s=1e-6, max_interval_s=1e-6,
                           measured_latency_s=1e-6)


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
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend([{"garbage": True}]),
                           shadow=False)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps == 0 and out.monitor_error

    def test_backend_exception_reaches_fail_safe(self):
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend([reading()]))
        ok, blockers = p.preflight()
        assert any("Astra" in b for b in blockers)


class TestShadowIsDefault:
    def test_shadow_on_by_default(self):
        assert PolicyPipeline("pi05_local_monitor").shadow is True
        assert MonitorGate().shadow is True

    def test_shadow_does_not_alter_the_executed_prefix(self):
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend(
                               [reading(progress="failed", execute_steps=0)]),
                           default_steps=5, shadow=True)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps == out.baseline_steps == 5
        assert out.gate["execute_steps"] < 5, "it WOULD have shortened"
        assert "SHADOW" in out.reason

    def test_gating_requires_explicit_switch(self):
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend(
                               [reading(progress="failed", execute_steps=0)]),
                           default_steps=5, shadow=False)
        out = p.step(state=STATE, proposed_steps=10)
        assert out.executed_steps != out.baseline_steps


class TestSchedulingStaysOutOfTheControlLoop:
    def test_unmeasured_rate_is_declared_dishonest(self):
        """The default cannot claim a rate it has not measured on the device."""
        ok, why = MonitorSchedule().honest()
        assert ok is False and "has not been measured" in why

    def test_measured_latency_sets_the_real_rate(self):
        s = MonitorSchedule.from_latency(6.0)
        assert s.honest()[0] is True
        assert s.effective_hz() == pytest.approx(1 / 6.0, abs=0.01)

    def test_a_rate_faster_than_inference_is_called_out(self):
        s = MonitorSchedule(min_interval_s=0.2, measured_latency_s=6.0)
        ok, why = s.honest()
        assert ok is False and "real rate is" in why

    def test_the_cost_of_a_slow_monitor_is_reported(self):
        d = MonitorSchedule.from_latency(6.0).to_log()
        assert d["control_cycles_between_looks"] == 1500, \
            "how many 250 Hz cycles pass between looks must be visible"

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
        ok, ev = d.due(state=closed, now=t + 2.0)
        assert ok and ev is Event.GRIPPER_CLOSE

    def test_a_gripper_close_during_inference_is_held_not_dropped(self):
        """The model cannot be re-entered mid-inference, but the event must
        still arrive. The gripper is on direct Modbus -- STOPFLAG does not
        stop it -- so a close nobody looked at is the dangerous case."""
        d = EventDetector()
        t = time.time()
        d.due(state=STATE, now=t)                       # first look consumes
        closed = list(STATE[:6]) + [1.0]
        ok, ev = d.due(state=closed, now=t + 0.3)       # inside min_interval
        assert (ok, ev) == (False, None), "cannot re-enter a busy model"
        ok, ev = d.due(state=closed, now=t + 1.2)       # next opportunity
        assert ok and ev is Event.GRIPPER_CLOSE, "the event must survive"
        assert d.deferred_events == 1, "the delay must be counted, not silent"

    def test_the_more_severe_held_event_wins(self):
        d = EventDetector()
        t = time.time()
        d.due(state=STATE, now=t)
        d.due(state=STATE, now=t + 0.1, in_approach_region=True)
        closed = list(STATE[:6]) + [1.0]
        d.due(state=closed, now=t + 0.2)
        ok, ev = d.due(state=closed, now=t + 1.2)
        assert ok and ev is Event.GRIPPER_CLOSE, \
            "approach must not displace a contact event"

    def test_stall_is_detected(self):
        d = EventDetector(MonitorSchedule(stall_cycles=3, min_interval_s=0.0))
        t = time.time()
        ev = None
        for i in range(6):
            _ok, e = d.due(state=STATE, now=t + i * 0.01)
            ev = e or ev
        assert ev in (Event.STALLED_MOTION, Event.PERIODIC)


class TestBackends:
    def test_one_model_everywhere_and_it_runs_on_the_jetson(self):
        from .vlm_backends import MONITOR_MODEL
        assert MONITOR_MODEL == "Qwen/Qwen3-VL-2B-Instruct"
        assert DEFAULT_MODEL == EDGE_MODEL == MONITOR_MODEL

    def test_the_retired_model_records_why_not_just_that(self):
        """It was retired by direction, not by measurement -- say so."""
        from .vlm_backends import RETIRED_EDGE_MODEL, RETIRED_REASON
        assert "SmolVLM2" in RETIRED_EDGE_MODEL
        assert "never evaluated on corrected inputs" in RETIRED_REASON

    def test_default_backend_refuses(self):
        b = UnconfiguredBackend()
        assert b.probe().available is False
        with pytest.raises(MonitorRejected):
            b.observe()

    def test_host_named_aliases_still_resolve(self):
        """a800/jetson named a MACHINE; the monitor runs locally on the Jetson
        and the A800 cannot serve it at all. Aliases keep callers working."""
        from .vlm_backends import DEPRECATED_KINDS
        for old in ("a800", "jetson", "edge"):
            assert DEPRECATED_KINDS[old] == "local"
            assert make_backend(old).model == DEFAULT_MODEL

    def test_probe_detects_a_missing_service(self):
        def boom(url, body, timeout):
            raise OSError("connection refused")
        b = OpenAICompatibleBackend(VlmConfig(), transport=boom)
        p = b.probe()
        assert p.available is False and "will not gate motion" in p.reason

    def test_same_instant_viewpoints_are_not_called_a_time_sequence(self):
        """The shadow-run defect: base+wrist from ONE tick were labelled
        oldest/current and the model was asked what changed. There is no honest
        answer, and SmolVLM2 correctly returned uncertain at zero confidence."""
        b = OpenAICompatibleBackend(VlmConfig())
        body = b.build_body(frames=["data:base", "data:wrist"], state_text="s",
                            intent_text="i")
        texts = [c["text"] for c in body["messages"][1]["content"]
                 if c["type"] == "text"]
        assert not any("oldest" in t for t in texts)
        assert any("same instant" in t for t in texts)
        assert any("not a time sequence" in t for t in texts)

    def test_labelled_pairs_carry_time_and_viewpoint(self):
        b = OpenAICompatibleBackend(VlmConfig(max_frames=4))
        body = b.build_body(
            frames=[("t-1 base", "a"), ("t-1 wrist", "b"),
                    ("t base", "c"), ("t wrist", "d")],
            state_text="s", intent_text="i")
        texts = [c["text"] for c in body["messages"][1]["content"]
                 if c["type"] == "text"]
        for tag in ("[t-1 base]", "[t-1 wrist]", "[t base]", "[t wrist]"):
            assert tag in texts
        assert not any("same instant" in t for t in texts)

    def test_prompt_tells_the_model_how_to_read_the_labels(self):
        from .vlm_backends import MONITOR_SYSTEM_PROMPT
        assert "same instant are different viewpoints" in MONITOR_SYSTEM_PROMPT
        assert "time labels differ" in MONITOR_SYSTEM_PROMPT

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
                           schedule=fast_schedule(),
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

    def test_metrics_surface_late_events(self):
        """Counting a deferral is pointless if nobody can see it."""
        p = PolicyPipeline("pi05_local_monitor", schedule=fast_schedule(),
                           backend=MockMonitorBackend([reading()] * 4))
        p.events.deferred_events = 7
        assert p.metrics()["events_deferred_by_monitor_latency"] == 7

    def test_metrics_report_clamping_and_streaks(self):
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
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
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(), backend=slow, shadow=True)
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
        p = PolicyPipeline("pi05_local_monitor_astra", schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
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
                           schedule=fast_schedule(),
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


class TestEscalationUsesTheRealAstraContract:
    """The defect these exist to prevent.

    The first implementation handed Astra a dict I invented -- {cycle, reason,
    reading, state, episode_id, task} -- while the real client reads
    packet["user_text"], packet["system"] and packet["response_schema"]. Plugged
    into the actual client that is a KeyError, which the pipeline caught and
    recorded as {"ok": false}. Escalations would have failed silently, with
    Astra never seeing a frame or a trajectory.

    The tests that let it through used `lambda pkt: {"ok": True}`, which accepts
    any shape. These use the REAL client instead, so a mismatch fails here.
    """

    def _pipeline(self, seen, *, answer=None):
        from .transports import AstraReviewSource

        def real_reviewer(pkt):
            seen.append(pkt)
            # The genuine client. If `pkt` is the wrong shape this raises.
            src = AstraReviewSource(base_url="https://unused", model="m",
                                    api_key_env="NOPE")
            src.build_body(pkt, pkt.get("image_data_urls"))
            return answer or {"ok": True,
                              "decision": {"mode": "student", "steps": 2}}

        return PolicyPipeline(
            "pi05_local_monitor_astra", schedule=fast_schedule(),
            backend=MockMonitorBackend([reading(progress="failed")] * 4),
            policy=MonitorPolicy(escalate_after_adverse=2,
                                 max_reading_age_s=60.0),
            astra_review=real_reviewer, shadow=False)

    def _escalate(self, p, **extra):
        t = time.time()
        for i in range(2):
            out = p.step(state=STATE, proposed_steps=10, now=t + i,
                         task="open the white dishwasher on the table",
                         episode_id="ep000000", **extra)
        return out

    def test_packet_is_accepted_by_the_real_client(self):
        seen = []
        out = self._escalate(self._pipeline(seen),
                             proposed_chunk=[[0.0] * 7] * 50)
        assert out.escalated is True
        assert len(seen) == 1
        assert out.astra_decision and out.astra_decision.get("ok") is True, \
            "a KeyError here would have been recorded as ok=false"

    def test_packet_carries_system_prompt_and_schema(self):
        seen = []
        self._escalate(self._pipeline(seen), proposed_chunk=[[0.0] * 7] * 50)
        pkt = seen[0]
        for key in ("system", "user_text", "response_schema"):
            assert key in pkt, f"the real client reads packet[{key!r}]"

    def test_packet_carries_the_proposed_trajectory(self):
        seen = []
        chunk = [[float(i)] * 7 for i in range(50)]
        self._escalate(self._pipeline(seen), proposed_chunk=chunk)
        assert "t+00" in seen[0]["user_text"], "the chunk must be visible"

    def test_packet_carries_the_frames(self):
        seen = []
        self._escalate(self._pipeline(seen), proposed_chunk=[[0.0] * 7] * 50,
                       frames_meta={"base": {"frame_index": 7,
                                             "frame_path": "/f/base.png",
                                             "frame_present": True}},
                       image_data_urls=["data:image/png;base64,AAA"])
        pkt = seen[0]
        assert "base" in pkt["frames"]
        assert pkt["image_data_urls"] == ["data:image/png;base64,AAA"]

    def test_packet_states_why_it_escalated(self):
        seen = []
        self._escalate(self._pipeline(seen), proposed_chunk=[[0.0] * 7] * 50)
        esc = seen[0]["escalation"]
        assert esc["cause"] and esc["monitor_reading"]
        assert esc["adverse_streak"] >= 2

    def test_a_wrong_shape_would_now_be_visible(self):
        """Guard on the guard: if the packet regressed, this class would fail."""
        from .transports import AstraReviewSource
        src = AstraReviewSource(base_url="https://x", model="m", api_key_env="N")
        with pytest.raises(KeyError):
            src.build_body({"cycle": 1, "reason": "x", "state": []}, None)


class TestAstraAnswerGoverns:
    """Second defect: Astra's answer only reached the log. While Astra 'had
    control' the monitor still decided how many steps ran, which made
    escalation a logging stub."""

    def _run(self, answer, *, proposed=10):
        p = PolicyPipeline(
            "pi05_local_monitor_astra", schedule=fast_schedule(),
            backend=MockMonitorBackend([reading(progress="failed")] * 4),
            policy=MonitorPolicy(escalate_after_adverse=2,
                                 max_reading_age_s=60.0),
            astra_review=lambda pkt: answer, shadow=False)
        t = time.time()
        for i in range(2):
            out = p.step(state=STATE, proposed_steps=proposed, now=t + i,
                         proposed_chunk=[[0.0] * 7] * 50, task="t",
                         episode_id="e")
        return p, out

    def test_astra_steps_are_executed_not_the_monitors(self):
        _p, out = self._run({"ok": True,
                             "decision": {"mode": "student", "steps": 4}})
        assert out.executed_steps == 4
        assert "astra reviewed and governs" in out.reason

    def test_astra_stop_executes_nothing(self):
        _p, out = self._run({"ok": True, "decision": {"mode": "stop"}})
        assert out.executed_steps == 0

    def test_astra_is_clamped_like_everyone_else(self):
        """A reviewer is not exempt from the bounds."""
        _p, out = self._run({"ok": True,
                             "decision": {"mode": "student", "steps": 999}},
                            proposed=6)
        assert out.executed_steps == 6, "bounded by proposed_steps"

    def test_unusable_astra_answer_falls_back_conservatively(self):
        _p, out = self._run({"ok": False, "error": "timeout"})
        assert "nothing usable" in out.reason
        assert out.executed_steps <= 3, "falls back to the monitor, not to permission"

    def test_malformed_astra_steps_are_not_trusted(self):
        _p, out = self._run({"ok": True,
                             "decision": {"mode": "student", "steps": "four"}})
        assert "nothing usable" in out.reason

    def test_unknown_astra_mode_is_not_trusted(self):
        _p, out = self._run({"ok": True,
                             "decision": {"mode": "teleport", "steps": 3}})
        assert "nothing usable" in out.reason

    def test_shadow_still_overrides_everything(self):
        p = PolicyPipeline(
            "pi05_local_monitor_astra", schedule=fast_schedule(),
            backend=MockMonitorBackend([reading(progress="failed")] * 4),
            policy=MonitorPolicy(escalate_after_adverse=2,
                                 max_reading_age_s=60.0),
            astra_review=lambda pkt: {"ok": True,
                                      "decision": {"mode": "student",
                                                   "steps": 9}},
            default_steps=5, shadow=True)
        t = time.time()
        for i in range(2):
            out = p.step(state=STATE, proposed_steps=10, now=t + i,
                         proposed_chunk=[[0.0] * 7] * 50)
        assert out.executed_steps == out.baseline_steps == 5


class TestShadowRunDefectsAreClosed:
    """The three wiring defects a SmolVLM2-500M shadow run exposed.

    Every evaluated tick came back uncertain/uncertain/unknown at confidence
    0.0. That was the model answering correctly: it was shown two camera
    viewpoints labelled as two moments, told the intent was "next 50 absolute
    joint targets" with no values, and handed the human demonstration instead
    of pi0.5's proposal. None of the three was evidence about the model.
    """

    def test_trajectory_is_rendered_not_counted(self):
        from .pipeline import describe_trajectory
        chunk = [[float(i)] * 7 for i in range(50)]
        text = describe_trajectory(chunk, [0.0] * 7)
        assert "t+00:" in text, "actual target values must be visible"
        assert "net displacement" in text
        assert text.strip() != "next 50 absolute joint targets"

    def test_gripper_transition_is_called_out(self):
        from .pipeline import describe_trajectory
        chunk = [[0.0] * 6 + [0.0]] * 25 + [[0.0] * 6 + [1.0]] * 25
        assert "closing" in describe_trajectory(chunk, [0.0] * 7)

    def test_missing_trajectory_says_so_rather_than_implying_one(self):
        from .pipeline import describe_trajectory
        text = describe_trajectory(None)
        assert "NO PROPOSED TRAJECTORY" in text and "uncertain" in text

    def test_preview_is_bounded_for_a_small_context(self):
        from .pipeline import INTENT_PREVIEW_STEPS, describe_trajectory
        chunk = [[float(i)] * 7 for i in range(50)]
        shown = describe_trajectory(chunk, [0.0] * 7).count("  t+")
        assert shown == INTENT_PREVIEW_STEPS <= 10

    def test_records_are_persisted_not_just_printed(self, tmp_path):
        """Shadow mode with no saved records cannot be evaluated afterwards."""
        rows = []
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend([reading()] * 3),
                           policy=MonitorPolicy(max_reading_age_s=60.0),
                           on_record=rows.append, shadow=True)
        t = time.time()
        for i in range(3):
            p.step(state=STATE, proposed_steps=5, now=t + i)
        assert len(rows) == 3
        for r in rows:
            for k in ("gate", "monitor_latency_s", "proposed_steps",
                      "executed_steps", "baseline_steps", "monitor_error"):
                assert k in r, k

    def test_persisted_record_carries_the_clamp_and_the_reason(self, tmp_path):
        rows = []
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend(
                               [reading(execute_steps=99)]),
                           policy=MonitorPolicy(max_reading_age_s=60.0),
                           on_record=rows.append, shadow=False)
        p.step(state=STATE, proposed_steps=50)
        g = rows[0]["gate"]
        assert g["clamped_from"] == 99
        assert g["execute_steps"] < 99 and g["reason"]


class TestAstraRoutingBugs:
    """Two bugs a Jetson run exposed: escalation was dead, and Astra ran every
    tick in every mode -- the triage design exactly inverted."""

    def test_pipeline_flags_a_missing_escalation_path(self):
        p = PolicyPipeline("pi05_local_monitor_astra",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend([reading()]))
        ok, blockers = p.preflight()
        assert ok is False
        assert any("Astra" in b for b in blockers), \
            "a dead escalation path must be visible in preflight"

    def test_local_monitor_mode_has_no_escalation_path_at_all(self):
        """Structural, not a matter of the caller declining to escalate."""
        p = PolicyPipeline("pi05_local_monitor",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend(
                               [reading(progress="failed")] * 6),
                           policy=MonitorPolicy(escalate_after_adverse=2,
                                                max_reading_age_s=60.0),
                           astra_review=None, shadow=False)
        t = time.time()
        for i in range(4):
            out = p.step(state=STATE, proposed_steps=10, now=t + i,
                         proposed_chunk=[[0.0] * 7] * 50)
        assert out.astra_called is False
        assert p.metrics()["astra_calls"] == 0

    def test_escalation_fires_exactly_once_not_per_tick(self):
        """The cost argument: Astra is reserved for persistence, not polled."""
        calls = []
        p = PolicyPipeline("pi05_local_monitor_astra",
                           schedule=fast_schedule(),
                           backend=MockMonitorBackend(
                               [reading(progress="failed")] * 6),
                           policy=MonitorPolicy(escalate_after_adverse=3,
                                                max_reading_age_s=60.0),
                           astra_review=lambda pkt: (calls.append(pkt),
                                                     {"ok": True,
                                                      "decision": {
                                                          "mode": "student",
                                                          "steps": 1}})[1],
                           shadow=False)
        t = time.time()
        for i in range(5):
            p.step(state=STATE, proposed_steps=10, now=t + i,
                   proposed_chunk=[[0.0] * 7] * 50)
        assert len(calls) == 1, \
            f"Astra must fire on escalation only, fired {len(calls)}x in 5 ticks"
        # and it stays one: sustained failure does not poll. See
        # TestAstraIsNotPolled, which this assertion originally uncovered.


class TestResponseIsCompactForLatency:
    def test_evidence_is_capped(self):
        from .vlm_monitor import EVIDENCE_MAX_CHARS, response_schema
        assert EVIDENCE_MAX_CHARS == 120
        assert response_schema()["properties"]["evidence"]["maxLength"] == 120

    def test_overlong_evidence_is_truncated_not_rejected(self):
        r = parse_reading(reading(evidence="x" * 5000))
        assert len(r.evidence) == 120

    def test_prompt_asks_for_a_short_clause(self):
        from .vlm_backends import MONITOR_SYSTEM_PROMPT
        assert "120 characters" in MONITOR_SYSTEM_PROMPT

    def test_jetson_config_fits_a_real_temporal_pair(self):
        """A pair means ONE camera at t-1 and t. Two cameras at one instant is
        not a time sequence, which is the defect this whole area came from.

        The default is 2 rather than 4 after measuring on the Jetson: each
        640x480 frame costs ~310 prompt tokens, and 4 frames made prefill ~6 s
        of a ~12 s budget. The CLI orders frames [cam t-1, cam t, ...] so the
        surviving pair after truncation is temporal, never two viewpoints.
        """
        from .vlm_backends import VlmConfig
        assert VlmConfig.jetson().max_frames >= 2, \
            "t-1 and t for one camera is two images"
        assert VlmConfig.jetson().max_frames % 2 == 0, \
            "an odd budget would truncate a pair to a single frame"

    def test_jetson_timeout_covers_the_measured_latency(self):
        """The 6 s starting point HAS now been measured and replaced.

        On the Jetson at MODE_30W (GPU 612 MHz, clocks pinned), Qwen3-VL-2B
        Q8_0 with a 2-frame pair took 5.5-7.0 s over 6 looks, and 9.1 s worst
        case over an earlier set. At 6 s every reading fail-safed to HOLD. The
        default must therefore exceed the measured worst case, with margin,
        and stay overridable for a device that measures differently."""
        from .vlm_backends import VlmConfig
        MEASURED_WORST_S = 9.1
        assert VlmConfig.jetson().timeout_s >= MEASURED_WORST_S, \
            "a timeout below the measured worst case fail-safes every cycle"
        assert VlmConfig.jetson(timeout_s=4.0).timeout_s == 4.0


class TestAstraIsNotPolled:
    """Found by a test whose expectation was wrong, which exposed a real gap:
    once escalated, EVERY subsequent adverse tick re-called Astra. That is
    polling an expensive reviewer for as long as things look bad -- the exact
    cost the local monitor exists to avoid."""

    def _run(self, recall, ticks=7):
        calls = []
        p = PolicyPipeline(
            "pi05_local_monitor_astra", schedule=fast_schedule(),
            backend=MockMonitorBackend([reading(progress="failed")] * (ticks + 2)),
            policy=MonitorPolicy(escalate_after_adverse=3,
                                 max_reading_age_s=60.0,
                                 astra_recall_every=recall),
            astra_review=lambda pkt: (calls.append(pkt),
                                      {"ok": True,
                                       "decision": {"mode": "student",
                                                    "steps": 1}})[1],
            shadow=False)
        t = time.time()
        for i in range(ticks):
            p.step(state=STATE, proposed_steps=10, now=t + i,
                   proposed_chunk=[[0.0] * 7] * 50)
        return p, calls

    def test_default_asks_once_on_the_transition(self):
        _p, calls = self._run(recall=0)
        assert len(calls) == 1, "sustained failure must not poll Astra"

    def test_a_cadence_can_be_configured_deliberately(self):
        _p, calls = self._run(recall=3)
        assert 1 < len(calls) <= 3

    def test_standing_decision_holds_between_asks(self):
        p, _calls = self._run(recall=0)
        later = [r for r in p.records if r.escalated][-1]
        assert later.astra_called is False, "not re-asked"
        assert later.executed_steps == 1, "Astra's standing answer still governs"

    def test_the_record_says_it_was_not_re_asked(self):
        p, _calls = self._run(recall=0)
        later = [r for r in p.records if r.escalated][-1]
        assert "not re-asked" in later.reason

    def test_default_policy_is_transition_only(self):
        assert MonitorPolicy().astra_recall_every == 0
