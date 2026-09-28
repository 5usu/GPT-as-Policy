"""One state machine, four modes, one audit format -- the KUKA loop, on YAM.

    observation -> pi0.5 proposal -> FK preview -> Astra review
      -> student/edit/eef decision -> deterministic sanitize -> YAM validation
      -> signed command envelope -> gateway -> feedback -> replan

The stages, the outcome vocabulary, the audit row and the per-mode ceiling
(replay and live_shadow stop at VALIDATE, structurally) are imported from
`kuka.loop` unchanged, so a YAM audit and a KUKA audit read line for line.
What differs is everything that knows the shape of a row: two arms, radians,
and a per-arm joint-space edit.

EXECUTION STATUS IS THE KUKA BRANCH'S: only a bounded student prefix is ever
emittable, edits are computed and recorded but not emitted, eef is refused,
and nothing reaches a real gateway from the CLI in this build.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Sequence

from ..kuka.loop import (FINAL_STAGE, AuditLog, CycleRecord, Outcome, Stage,
                         _loggable_observation)
from .contract import (ACTION_DIM, ARM_JOINT_INDICES, ARMS, CONTROL_HZ,
                       JOINT_INDICES, JOINTS_PER_ARM, MAX_CORRECTED_STEPS,
                       MAX_CORRECTION_RAD, MAX_STUDENT_STEPS, ROBOT_MODEL,
                       eef_execution_gate)
from .experiment import assess_success, retry_limits, stop_conditions
from .kinematics import UnavailableFK
from .packet import build_packet
from .safety import (ArmingRefused, CommandLedger, Mode, RigIdentity, Supervisor,
                     authorise, require_command_capable)
from .sanitize import sanitize
from .validation import (execution_eligible, has_fatal, summarize, validate_chunk,
                         worsens)

SCHEMA = "hybrid_rollout.robodojo.yam.loop.v1"

__all__ = ["AuditLog", "EDIT_EXECUTION_ENABLED", "EXECUTABLE_DECISION_MODES",
           "Outcome", "Stage", "YamReviewLoop", "apply_edit",
           "evaluate_stop_signals"]

EXECUTABLE_DECISION_MODES = frozenset({"student"})
EDIT_EXECUTION_ENABLED = False
EDIT_LOCK_REASON = (
    "per-arm joint edits are deliverable to the YAM driver, but execution of a "
    "reviewer correction stays disabled until a supervised campaign validates "
    "it. The edit is still computed, validated and recorded; it is not emitted.")


def evaluate_stop_signals(*, configured: Sequence[str], observation: dict[str, Any],
                          feedback: dict[str, Any] | None,
                          tolerance_rad: float | None) -> list[str]:
    """Which configured stop conditions hold. Unevaluable is reported, not passed."""
    flags = {"unexpected_contact_or_load": "load_exceeded",
             "outside_workspace": "outside_workspace",
             "stale_vision": "vision_stale",
             "heartbeat_lost": "heartbeat_lost",
             "estop_engaged": "estop_engaged"}
    hit: list[str] = []
    for cond in configured:
        if cond in flags and observation.get(flags[cond]):
            hit.append(cond)
        elif cond == "commanded_observed_disagreement" and feedback and feedback.get("ok"):
            if tolerance_rad is None:
                hit.append(f"{cond}:unevaluable")
                continue
            cmd = feedback.get("commanded") or []
            obs = feedback.get("measured") or []
            if len(cmd) == len(obs) == ACTION_DIM:
                worst = max(abs(cmd[j] - obs[j]) for j in JOINT_INDICES)
                if worst > float(tolerance_rad):
                    hit.append(cond)
    return hit


def apply_edit(rows: Sequence[Sequence[float]], edit: dict[str, Any],
               n_steps: int) -> tuple[list[list[float]] | None, str]:
    """Apply a per-arm joint-space edit to the leading `n_steps` rows.

    Returns (edited_rows, "") or (None, refusal). An arm with no entry is left
    alone. The per-arm gripper request is RECORDED by the caller, not applied:
    as on the KUKA, the edit channel carries joint corrections only.
    """
    edited = [list(r) for r in rows[:n_steps]]
    for arm in ARMS:
        spec = (edit or {}).get(arm)
        if spec is None:
            continue
        deltas = spec.get("delta_joint_rad") if isinstance(spec, dict) else None
        if not isinstance(deltas, list) or len(deltas) != JOINTS_PER_ARM:
            return None, f"{arm}.delta_joint_rad must have {JOINTS_PER_ARM} values"
        for j, x in enumerate(deltas):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None, f"{arm} delta {j} is not numeric"
            if abs(x) > MAX_CORRECTION_RAD + 1e-12:
                return None, (f"{arm} delta {j} = {x:+.5f} exceeds the "
                              f"+/-{MAX_CORRECTION_RAD:.5f} rad bound")
        for row in edited:
            for k, idx in enumerate(ARM_JOINT_INDICES[arm]):
                row[idx] += float(deltas[k])
    return edited, ""


class YamReviewLoop:
    def __init__(self, *, mode: Mode, config: dict[str, Any],
                 raw_config: dict[str, Any] | None = None,
                 proposal_source=None, review_source=None, gateway=None,
                 fk=None, audit: AuditLog | None = None,
                 target: RigIdentity | None = None,
                 allowlist: Sequence[RigIdentity] = (),
                 supervisor: Supervisor | None = None,
                 ledger: CommandLedger | None = None,
                 secret: bytes | None = None,
                 hz: float = CONTROL_HZ,
                 task_reference: Any = None) -> None:
        self.mode = mode
        self.config = dict(config)
        self.raw_config = dict(raw_config or {})
        self.proposal_source = proposal_source
        self.review_source = review_source
        self.gateway = gateway
        self.fk = fk or UnavailableFK()
        self.audit = audit
        self.target = target
        self.allowlist = list(allowlist)
        self.supervisor = supervisor or Supervisor()
        self.ledger = ledger or CommandLedger()
        self.secret = secret
        self.hz = hz
        self.task_reference = task_reference
        self.cycle = 0
        self.retries = 0
        self.stopped = False
        self.stop_reason = ""

    @property
    def commands_possible(self) -> bool:
        try:
            require_command_capable(self.mode)
        except ArmingRefused:
            return False
        return self.gateway is not None

    def _finish(self, rec: CycleRecord, stage: Stage, outcome: Outcome,
                reason: str) -> CycleRecord:
        rec.stage_reached = stage.value
        rec.outcome = outcome.value
        rec.reason = reason
        rec.commands_possible_in_mode = FINAL_STAGE.get(self.mode) is Stage.REPLAN
        if self.audit:
            self.audit.append(rec)
        return rec

    def step(self, observation: dict[str, Any], *, now: float | None = None,
             camera_assessment=None, supervisor_confirmed: bool | None = None,
             predicate_value: Any = None) -> CycleRecord:
        now = time.time() if now is None else now
        self.cycle += 1
        rec = CycleRecord(schema=SCHEMA, robot_model=ROBOT_MODEL,
                          mode=self.mode.value, cycle=self.cycle,
                          observation=_loggable_observation(observation))
        limits = retry_limits(self.raw_config)
        conds = stop_conditions(self.raw_config)
        tol = (self.raw_config.get("stop_conditions") or {}).get(
            "commanded_observed_tolerance_rad")
        tol = None if tol in ("", None) else float(tol)

        signals = evaluate_stop_signals(configured=conds, observation=observation,
                                        feedback=None, tolerance_rad=tol)
        rec.stop_signals = signals
        if signals or self.stopped:
            self.stopped = True
            self.stop_reason = self.stop_reason or ", ".join(signals)
            return self._finish(rec, Stage.OBSERVE, Outcome.STOPPED,
                                f"controlled stop: {self.stop_reason}")

        rec.success = assess_success(
            camera=camera_assessment, predicate_value=predicate_value,
            predicate_config=self.config.get("success_predicate"),
            supervisor_confirmed=supervisor_confirmed)

        # --- PROPOSE ----------------------------------------------------------
        if self.proposal_source is None:
            return self._finish(rec, Stage.PROPOSE, Outcome.NO_PROPOSAL,
                                "no proposal source configured")
        prop = self.proposal_source.propose(observation)
        if not prop.get("ok"):
            return self._finish(rec, Stage.PROPOSE, Outcome.NO_PROPOSAL,
                                str(prop.get("error", "proposal failed")))
        rows = [list(r) for r in prop["rows"]]
        rec.proposal = {"source": prop.get("source"),
                        "is_live": bool(prop.get("is_live")),
                        "checkpoint_id": prop.get("checkpoint_id"),
                        "n_steps": len(rows), "values": rows}

        # --- FK PREVIEW -------------------------------------------------------
        rec.fk_preview = self.fk.preview(rows).to_log()

        # --- REVIEW -----------------------------------------------------------
        state = observation.get("state")
        before = validate_chunk(rows, state=state, hz=self.hz)
        rec.violations_before = summarize(before)
        if has_fatal(before):
            return self._finish(rec, Stage.VALIDATE, Outcome.HELD,
                                "proposal has FATAL violations; nothing emitted")
        if self.review_source is None:
            return self._finish(rec, Stage.REVIEW, Outcome.NO_DECISION,
                                "no review source configured")
        provenance = (prop.get("provenance")
                      or getattr(self.proposal_source, "provenance", None)
                      or "model_predicted")
        if provenance == "recorded_chunk_replay":
            provenance = "model_predicted"
        try:
            packet = build_packet(
                task_instruction=str(observation.get("task", "")),
                observation_id=str(observation.get("observation_id", "")),
                state=state, chunk=rows, provenance=provenance,
                frames=observation.get("frames") or {}, fk_preview=rec.fk_preview,
                reference=self.task_reference)
        except (TypeError, ValueError, IndexError) as exc:
            return self._finish(rec, Stage.REVIEW, Outcome.NO_DECISION,
                                f"could not build review packet: {exc}")
        packet["mode"] = self.mode.value
        packet["image_data_urls"] = list(observation.get("image_data_urls") or [])
        rv = self.review_source.review(packet)
        rec.review = {"source": getattr(self.review_source, "name", "?"),
                      "is_live": bool(getattr(self.review_source, "is_live", False)),
                      "ok": bool(rv.get("ok")), "error": rv.get("error"),
                      "dry_run": bool(rv.get("dry_run")), "usage": rv.get("usage"),
                      "attempts_used": rv.get("attempts_used")}
        if not rv.get("ok"):
            return self._finish(rec, Stage.REVIEW, Outcome.NO_DECISION,
                                str(rv.get("error", "review failed")))
        decision = dict(rv["decision"])
        rec.decision = decision
        dmode = str(decision.get("mode", "student"))

        # --- DECIDE -----------------------------------------------------------
        if dmode == "stop":
            self.stopped = True
            self.stop_reason = "reviewer requested stop"
            return self._finish(rec, Stage.DECIDE, Outcome.STOPPED,
                                f"reviewer stop: {str(decision.get('reason', ''))[:160]}")

        requested = int(decision.get("steps") or 1)
        n_student = max(1, min(requested, MAX_STUDENT_STEPS, len(rows)))
        candidate = [list(r) for r in rows[:n_student]]
        outcome = Outcome.STUDENT

        if dmode == "eef":
            allowed, missing, why = eef_execution_gate()
            rec.decision = {**decision, "eef_gate": {
                "accepted_as_upstream_decision": True,
                "static_gate_allowed": allowed, "missing_prerequisites": missing,
                "execution_allowed": False, "reason": why}}
            return self._finish(rec, Stage.DECIDE, Outcome.EEF_REFUSED, why)

        if dmode not in ("student", "edit"):
            return self._finish(rec, Stage.DECIDE, Outcome.GATE_BLOCKED,
                                f"decision mode {dmode!r} is not supported on YAM")

        if dmode == "edit":
            n_edit = max(1, min(requested, MAX_CORRECTED_STEPS, len(rows)))
            edited, refusal = apply_edit(rows, decision.get("edit") or {}, n_edit)
            grip_req = {arm: ((decision.get("edit") or {}).get(arm) or {}).get("gripper")
                        for arm in ARMS}
            if edited is None:
                rec.improvement_valid = False
                outcome = Outcome.REJECTED_EDIT
                rec.reason = refusal
            else:
                seg = [list(r) for r in rows[:n_edit]]
                vb = validate_chunk(seg, state=state, hz=self.hz)
                va = validate_chunk(edited, state=state, hz=self.hz)
                bad, why = worsens(vb, va)
                if bad or has_fatal(va):
                    rec.improvement_valid = False
                    outcome = Outcome.REJECTED_EDIT
                    rec.reason = why
                elif not EDIT_EXECUTION_ENABLED:
                    rec.improvement_valid = True
                    rec.decision = {**decision, "edit_computed": {
                        "accepted": True, "n_steps": n_edit,
                        "gripper_requested": grip_req, "gripper_applied": False,
                        "emitted": False, "lock_reason": EDIT_LOCK_REASON}}
                    outcome = Outcome.EDIT_LOCKED
                else:
                    rec.improvement_valid = True
                    candidate, outcome = edited, Outcome.APPLIED_EDIT

        # --- SANITIZE + VALIDATE ----------------------------------------------
        final, srep = sanitize(candidate, state=state, hz=self.hz)
        rec.sanitizer = srep.to_log()
        after = validate_chunk(final, state=state, hz=self.hz)
        rec.violations_after = summarize(after)
        safe, blockers = execution_eligible(after, bool(final))
        rec.execution_safe = safe
        rec.execution_blockers = blockers
        rec.replan = {"next_from": "recorded dataset" if self.mode is Mode.REPLAY
                      else "next supplied observation",
                      "caused_by_this_proposal": False}

        if FINAL_STAGE[self.mode] is Stage.VALIDATE:
            return self._finish(
                rec, Stage.VALIDATE, Outcome.OBSERVED_ONLY,
                f"{self.mode.value}: review complete; command construction is not "
                f"reachable in this mode ({outcome.value}, execution_safe={safe})")

        emit_mode = "edit" if outcome is Outcome.APPLIED_EDIT else "student"
        rec.emitted_mode = emit_mode
        if emit_mode not in EXECUTABLE_DECISION_MODES:
            return self._finish(rec, Stage.VALIDATE, Outcome.GATE_BLOCKED,
                                f"emitting {emit_mode!r} is not permitted here")

        # --- ENVELOPE ---------------------------------------------------------
        one_step = [list(final[0])] if final else []
        try:
            env = authorise(
                mode=self.mode, config=self.config, target=self.target,
                allowlist=self.allowlist, supervisor=self.supervisor,
                ledger=self.ledger, command_id=str(uuid.uuid4()),
                rows=one_step, control_hz=self.hz,
                observation_id=str(observation.get("observation_id", "")),
                observation_epoch=observation.get("epoch"),
                decision_mode=dmode,
                audit={"cycle": self.cycle, "outcome": outcome.value,
                       "violations_after": rec.violations_after["codes"],
                       "sanitizer_changed": bool(rec.sanitizer.get("changes_emitted"))},
                secret=self.secret, now=now, execution_safe=safe)
        except ArmingRefused as exc:
            rec.envelope = {"authorised": False, "code": exc.code,
                            "detail": exc.detail, "missing": exc.missing}
            return self._finish(rec, Stage.ENVELOPE, Outcome.ARMING_REFUSED,
                                f"{exc.code}: {exc.detail}")
        rec.envelope = env.to_log()
        rec.approved_for_execution = env.approved_for_execution

        # --- GATEWAY + FEEDBACK -----------------------------------------------
        if self.gateway is None:
            return self._finish(rec, Stage.GATEWAY, Outcome.ARMING_REFUSED,
                                "no gateway configured")
        res = self.gateway.send(env)
        rec.gateway = {"name": getattr(self.gateway, "name", "?"),
                       "can_move_robot": bool(getattr(self.gateway, "can_move_robot", False)),
                       "sent": bool(res.get("sent")), "shadow": bool(res.get("shadow"))}
        fb = None
        if hasattr(self.gateway, "feedback"):
            fb = self.gateway.feedback()
            if fb and fb.get("ok"):
                fb = {**fb, "commanded": list(one_step[0])}
        rec.feedback = fb
        post = evaluate_stop_signals(configured=conds, observation=observation,
                                     feedback=fb, tolerance_rad=tol)
        if post:
            rec.stop_signals = sorted(set(rec.stop_signals) | set(post))
            self.stopped = True
            self.stop_reason = ", ".join(post)
            return self._finish(rec, Stage.FEEDBACK, Outcome.STOPPED,
                                f"controlled stop after feedback: {self.stop_reason}")
        if outcome is Outcome.REJECTED_EDIT:
            self.retries += 1
            if self.retries > limits["max_retries_per_subgoal"]:
                self.stopped = True
                self.stop_reason = "retry limit exceeded"
                return self._finish(rec, Stage.REPLAN, Outcome.STOPPED,
                                    f"retry limit {limits['max_retries_per_subgoal']} exceeded")
        return self._finish(rec, Stage.REPLAN, Outcome.COMMAND_SENT,
                            f"one supervised step delivered ({outcome.value})")
