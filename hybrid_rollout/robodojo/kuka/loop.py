"""One state machine, four modes, one audit format.

    observation -> pi0.5 proposal -> FK preview -> Astra review
      -> student/edit/eef decision -> deterministic sanitize -> KUKA validation
      -> signed command envelope -> Jetson RSI gateway -> feedback -> replan

The modes differ ONLY in where proposals come from and how far down the chain a
cycle is permitted to travel. They share the state machine, the gates and the
audit row, so a shadow run and an execution run are comparable line for line.

  replay             recorded frames + recorded trajectory -> review. Stops after
                     VALIDATE. No envelope is ever constructed.
  live_shadow        live frames -> progress/success prediction. Stops after
                     VALIDATE. Commands suppressed.
  reviewed_execution pi0.5 proposes, Astra decides, one supervised step.
  astra_direct       Astra proposes its own bounded action. IDENTICAL gates plus
                     four extra prerequisites, and it is meant to be hard to arm.

`eef` is accepted as an upstream decision in every mode and refused at the KUKA
execution gate; it never becomes a command.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .contract import (ACTION_DIM, ARM_DIM, CONTROL_HZ, MAX_CORRECTED_STEPS,
                       MAX_CORRECTION_DEG, MAX_GRIPPER_DELTA, MAX_STUDENT_STEPS,
                       ROBOT_MODEL, eef_execution_gate)
from .eef import CartesianCapability, eef_eligible, validate_target
from .interfaces import cartesian_capability, modes_locked
from .experiment import CameraAssessment, assess_success, retry_limits, stop_conditions
from .kinematics import UnavailableFK
from .safety import (ArmingRefused, CommandLedger, Mode, RobotIdentity, Supervisor,
                     authorise, missing_config, require_command_capable)
from .sanitize import sanitize
from .validation import (execution_eligible, has_fatal, summarize, validate_chunk,
                         worsens)

SCHEMA = "hybrid_rollout.robodojo.kuka.loop.v1"


class Stage(str, Enum):
    OBSERVE = "observe"
    PROPOSE = "propose"
    FK_PREVIEW = "fk_preview"
    REVIEW = "review"
    DECIDE = "decide"
    SANITIZE = "sanitize"
    VALIDATE = "validate"
    ENVELOPE = "envelope"
    GATEWAY = "gateway"
    FEEDBACK = "feedback"
    REPLAN = "replan"


#: VERIFIED DEPLOYMENT FACT. The deployed RSI receive configuration accepts
#: AK.A1..A6 plus STOPFLAG only. Joint corrections ARE deliverable over that, so
#: `edit` is not blocked by the interface -- but it stays off until a supervised
#: campaign has validated it. Only a bounded STUDENT PREFIX of pi0.5 joint
#: actions may execute for now. `eef` and Astra-direct are blocked by the
#: interface itself and cannot be enabled by configuration here.
#: `astra_direct_joint` is executable in principle -- this controller accepts
#: AK.A1..A6, so a self-proposed joint action is deliverable. It is NOT enabled
#: by default: `ReviewLoop(direct_bounds=...)` must be given validated bounds,
#: and every other gate still applies. Cartesian `eef`/`astra_direct` remain
#: blocked by the interface and no configuration here changes that.
EXECUTABLE_DECISION_MODES = frozenset({"student", "astra_direct_joint"})
EDIT_EXECUTION_ENABLED = False
EDIT_LOCK_REASON = (
    "joint edits are deliverable over AK.A1-A6, but execution of a reviewer "
    "correction stays disabled until a supervised campaign validates it. The "
    "edit is still computed, validated and recorded; it is simply not emitted.")


class Outcome(str, Enum):
    OBSERVED_ONLY = "observed_only"          # replay / shadow completed a review
    STUDENT = "student"
    APPLIED_EDIT = "applied_edit"
    REJECTED_EDIT = "rejected_edit"
    EEF_REFUSED = "eef_refused"
    EDIT_LOCKED = "edit_locked"
    GATE_BLOCKED = "gate_blocked"
    ARMING_REFUSED = "arming_refused"
    COMMAND_SENT = "command_sent"
    STOPPED = "stopped"
    HELD = "held"
    NO_PROPOSAL = "no_proposal"
    NO_DECISION = "no_decision"


#: How far each mode may travel. Structural, checked before anything is built.
FINAL_STAGE: dict[Mode, Stage] = {
    Mode.REPLAY: Stage.VALIDATE,
    Mode.LIVE_SHADOW: Stage.VALIDATE,
    Mode.REVIEWED_EXECUTION: Stage.REPLAN,
    Mode.ASTRA_DIRECT: Stage.REPLAN,
}


@dataclass
class CycleRecord:
    """One cycle of the state machine. The shared audit row for all four modes."""
    schema: str = SCHEMA
    robot_model: str = ROBOT_MODEL
    mode: str = ""
    cycle: int = 0
    stage_reached: str = Stage.OBSERVE.value
    outcome: str = ""
    reason: str = ""
    observation: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    fk_preview: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    sanitizer: dict[str, Any] | None = None
    violations_before: dict[str, Any] = field(default_factory=dict)
    violations_after: dict[str, Any] = field(default_factory=dict)
    improvement_valid: bool | None = None
    execution_safe: bool = False
    execution_blockers: list[str] = field(default_factory=list)
    envelope: dict[str, Any] | None = None
    gateway: dict[str, Any] | None = None
    feedback: dict[str, Any] | None = None
    success: dict[str, Any] | None = None
    stop_signals: list[str] = field(default_factory=list)
    replan: dict[str, Any] = field(default_factory=dict)
    emitted_mode: str | None = None      # what actually leaves, not what was asked
    commands_possible_in_mode: bool = False
    approved_for_execution: bool = False
    written_at: str = ""

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return d


class AuditLog:
    """Append-only JSONL. Never truncates, never rewrites."""

    def __init__(self, path: str | Path) -> None:
        import json
        self._json = json
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def append(self, rec: CycleRecord) -> dict[str, Any]:
        row = rec.to_log()
        with self.path.open("a") as f:        # "a" only
            f.write(self._json.dumps(row) + "\n")
        self.n += 1
        return row


def evaluate_stop_signals(*, configured: Sequence[str], observation: dict[str, Any],
                          feedback: dict[str, Any] | None,
                          tolerance_deg: float | None) -> list[str]:
    """Which configured stop conditions currently hold.

    A condition that cannot be evaluated (because its limit was never supplied)
    is reported as `<name>:unevaluable` rather than silently passing.
    """
    hit: list[str] = []
    for cond in configured:
        if cond == "lost_handle" and observation.get("handle_visible") is False:
            hit.append(cond)
        elif cond == "unexpected_contact_or_load" and observation.get("load_exceeded"):
            hit.append(cond)
        elif cond == "door_outside_expected_region" and observation.get("door_outside_region"):
            hit.append(cond)
        elif cond == "stale_vision" and observation.get("vision_stale"):
            hit.append(cond)
        elif cond == "heartbeat_lost" and observation.get("heartbeat_lost"):
            hit.append(cond)
        elif cond == "estop_engaged" and observation.get("estop_engaged"):
            hit.append(cond)
        elif cond == "commanded_observed_disagreement":
            if feedback and feedback.get("ok"):
                if tolerance_deg is None:
                    hit.append(f"{cond}:unevaluable")
                else:
                    cmd = feedback.get("commanded") or []
                    obs = feedback.get("measured") or []
                    if cmd and obs and len(cmd) == len(obs):
                        worst = max(abs(a - b) for a, b in
                                    zip(cmd[:ARM_DIM], obs[:ARM_DIM]))
                        if worst > float(tolerance_deg):
                            hit.append(cond)
    return hit


class KukaReviewLoop:
    def __init__(self, *, mode: Mode, config: dict[str, Any],
                 raw_config: dict[str, Any] | None = None,
                 proposal_source=None, review_source=None, gateway=None,
                 fk=None, audit: AuditLog | None = None,
                 target: RobotIdentity | None = None,
                 allowlist: Sequence[RobotIdentity] = (),
                 supervisor: Supervisor | None = None,
                 ledger: CommandLedger | None = None,
                 secret: bytes | None = None,
                 hz: float = CONTROL_HZ,
                 cartesian: CartesianCapability | None = None,
                 direct_bounds: Any = None) -> None:
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
        self.cartesian = cartesian or CartesianCapability()
        self.direct_bounds = direct_bounds
        self.cycle = 0
        self.retries = 0
        self.no_progress = 0
        self.stopped = False
        self.stop_reason = ""

    # ------------------------------------------------------------------ helpers
    @property
    def commands_possible(self) -> bool:
        try:
            require_command_capable(self.mode)
        except ArmingRefused:
            return False
        return bool(self.gateway is not None)

    def _finish(self, rec: CycleRecord, stage: Stage, outcome: Outcome,
                reason: str) -> CycleRecord:
        rec.stage_reached = stage.value
        rec.outcome = outcome.value
        rec.reason = reason
        rec.commands_possible_in_mode = self.mode in FINAL_STAGE and \
            FINAL_STAGE[self.mode] is Stage.REPLAN
        if self.audit:
            self.audit.append(rec)
        return rec

    # --------------------------------------------------------------- one cycle
    def step(self, observation: dict[str, Any], *,
             now: float | None = None,
             camera_assessment: CameraAssessment | None = None,
             supervisor_confirmed: bool | None = None,
             predicate_value: Any = None) -> CycleRecord:
        now = time.time() if now is None else now
        self.cycle += 1
        rec = CycleRecord(mode=self.mode.value, cycle=self.cycle,
                          observation=dict(observation))
        limits = retry_limits(self.raw_config)
        conds = stop_conditions(self.raw_config)
        tol = (self.raw_config.get("stop_conditions") or {}).get(
            "commanded_observed_tolerance_deg")
        tol = None if tol in ("", None) else float(tol)

        # --- stop conditions are evaluated FIRST, every cycle -----------------
        signals = evaluate_stop_signals(configured=conds, observation=observation,
                                        feedback=None, tolerance_deg=tol)
        rec.stop_signals = signals
        if signals or self.stopped:
            self.stopped = True
            self.stop_reason = self.stop_reason or ", ".join(signals)
            return self._finish(rec, Stage.OBSERVE, Outcome.STOPPED,
                                f"controlled stop: {self.stop_reason}")

        # --- success assessment is recorded in every mode ---------------------
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
        rec.fk_preview = self.fk.preview([r[:ARM_DIM] for r in rows]).to_log()

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
        packet = {"request_id": observation.get("observation_id"),
                  "mode": self.mode.value}
        rv = self.review_source.review(packet)
        rec.review = {"source": getattr(self.review_source, "name", "?"),
                      "is_live": bool(getattr(self.review_source, "is_live", False)),
                      "ok": bool(rv.get("ok")), "error": rv.get("error")}
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
                                f"reviewer stop: {str(decision.get('reason',''))[:160]}")

        requested = int(decision.get("steps") or 1)
        n_student = max(1, min(requested, MAX_STUDENT_STEPS, len(rows)))
        candidate = [list(r) for r in rows[:n_student]]
        outcome = Outcome.STUDENT

        if dmode == "eef":
            # Two independent gates, both fail-closed. The static one asks
            # whether this build may ever do Cartesian; the capability one asks
            # whether THIS cell has attested the facts that make it checkable.
            allowed, missing, why = eef_execution_gate()
            cap_ok, cap_missing = self.cartesian.allowed()
            viol = []
            if allowed and cap_ok:
                viol = validate_target(
                    decision.get("target") or {},
                    measured=observation.get("measured_pose") or {},
                    capability=self.cartesian)
            v_ok, v_codes = eef_eligible(viol)
            rec.decision = {**decision, "eef_gate": {
                "accepted_as_upstream_decision": True,
                "static_gate_allowed": allowed, "missing_prerequisites": missing,
                "cell_attested": cap_ok, "cell_missing": cap_missing,
                "target_violations": [x.to_log() for x in viol],
                "execution_allowed": bool(allowed and cap_ok and v_ok),
                "reason": why,
                "resolver_note": ("a Cartesian target is resolved by the "
                                  "CONTROLLER; joint angles cannot be checked "
                                  "here before they exist")}}
            if not (allowed and cap_ok and v_ok):
                detail = why if not allowed else (
                    "cell not attested: " + ", ".join(cap_missing) if not cap_ok
                    else "target rejected: " + ", ".join(v_codes))
                return self._finish(rec, Stage.DECIDE, Outcome.EEF_REFUSED, detail)

        if dmode == "astra_direct_joint":
            from .astra_direct import direct_eligible, validate_direct
            rows_d = decision.get("joint_targets_deg") or []
            if self.direct_bounds is None:
                return self._finish(
                    rec, Stage.DECIDE, Outcome.GATE_BLOCKED,
                    "astra_direct_joint proposed but no validated bounds were "
                    "supplied. Direct proposal removes the policy's sanity floor, "
                    "so bounds are required, not optional.")
            viol = validate_direct(rows_d, measured=state, bounds=self.direct_bounds)
            ok_d, codes = direct_eligible(viol)
            rec.decision = {**decision, "direct_gate": {
                "bounds": self.direct_bounds.to_log(),
                "violations": [x.to_log() for x in viol],
                "accepted": ok_d}}
            if not ok_d:
                return self._finish(rec, Stage.DECIDE, Outcome.GATE_BLOCKED,
                                    f"astra_direct_joint rejected: {', '.join(codes)}")
            candidate = [list(r) + [state[ARM_DIM]] if len(r) == ARM_DIM else list(r)
                         for r in rows_d]
            outcome = Outcome.APPLIED_EDIT
            rec.improvement_valid = True

        if dmode == "edit":
            edit = decision.get("edit") or {}
            deltas = edit.get("delta_joint_deg")
            n_edit = max(1, min(requested, MAX_CORRECTED_STEPS, len(rows)))
            refusal = ""
            if not isinstance(deltas, list) or len(deltas) != ARM_DIM:
                refusal = f"delta_joint_deg must have {ARM_DIM} values"
            else:
                for j, x in enumerate(deltas):
                    if isinstance(x, bool) or not isinstance(x, (int, float)):
                        refusal = f"delta {j} is not numeric"; break
                    if abs(x) > MAX_CORRECTION_DEG + 1e-9:
                        refusal = (f"delta {j} = {x:+.4f} exceeds the "
                                   f"+/-{MAX_CORRECTION_DEG} deg bound"); break
            if refusal:
                rec.improvement_valid = False
                candidate, outcome = [list(r) for r in rows[:n_student]], Outcome.REJECTED_EDIT
                rec.reason = refusal
            else:
                edited = [list(r) for r in rows[:n_edit]]
                for i in range(len(edited)):
                    for j in range(ARM_DIM):
                        edited[i][j] += float(deltas[j])
                seg = [list(r) for r in rows[:n_edit]]
                vb = validate_chunk(seg, state=state, hz=self.hz)
                va = validate_chunk(edited, state=state, hz=self.hz)
                bad, why = worsens(vb, va)
                if bad or has_fatal(va):
                    rec.improvement_valid = False
                    candidate, outcome = [list(r) for r in rows[:n_student]], Outcome.REJECTED_EDIT
                    rec.reason = why
                elif not EDIT_EXECUTION_ENABLED:
                    # The edit was sound. It is still not emitted: only a
                    # student prefix may execute in this build. Recorded in
                    # full so the campaign that unlocks it has evidence.
                    rec.improvement_valid = True
                    rec.decision = {**decision, "edit_computed": {
                        "accepted": True, "n_steps": n_edit,
                        "delta_joint_deg": list(deltas),
                        "emitted": False, "lock_reason": EDIT_LOCK_REASON}}
                    candidate = [list(r) for r in rows[:n_student]]
                    outcome = Outcome.EDIT_LOCKED
                else:
                    rec.improvement_valid = True
                    candidate, outcome = edited, Outcome.APPLIED_EDIT

        # --- SANITIZE ---------------------------------------------------------
        final, srep = sanitize(candidate, state=state, hz=self.hz)
        rec.sanitizer = srep.to_log()

        # --- VALIDATE ---------------------------------------------------------
        after = validate_chunk(final, state=state, hz=self.hz)
        rec.violations_after = summarize(after)
        safe, blockers = execution_eligible(after, bool(final))
        rec.execution_safe = safe
        rec.execution_blockers = blockers

        rec.replan = {"next_from": "recorded dataset" if self.mode is Mode.REPLAY
                      else "next supplied observation",
                      "caused_by_this_proposal": False,
                      "statement": ("The next observation is supplied from outside "
                                    "this loop. In replay and live_shadow nothing "
                                    "was executed, so no observation is a "
                                    "consequence of any reviewed proposal.")}

        # --- observation-only modes stop here, structurally ------------------
        if FINAL_STAGE[self.mode] is Stage.VALIDATE:
            return self._finish(
                rec, Stage.VALIDATE, Outcome.OBSERVED_ONLY,
                f"{self.mode.value}: review complete; command construction is not "
                f"reachable in this mode ({outcome.value}, execution_safe={safe})")

        # --- MODE LOCK: gate on what is EMITTED, not what was requested ------
        # A rejected or locked edit falls back to the unmodified student prefix,
        # and that prefix is exactly what this build permits. Gating on the
        # reviewer's requested mode would block a fallback that is already the
        # permitted thing -- the question is what leaves this function.
        emit_mode = ("astra_direct_joint" if dmode == "astra_direct_joint"
                     else "edit" if outcome is Outcome.APPLIED_EDIT else "student")
        rec.emitted_mode = emit_mode
        if emit_mode not in EXECUTABLE_DECISION_MODES:
            locked = modes_locked().get(emit_mode, "not executable in this build")
            return self._finish(
                rec, Stage.VALIDATE, Outcome.GATE_BLOCKED,
                f"emitting {emit_mode!r} is not permitted here: {locked}")

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

        # --- GATEWAY ----------------------------------------------------------
        if self.gateway is None:
            return self._finish(rec, Stage.GATEWAY, Outcome.ARMING_REFUSED,
                                "no gateway configured")
        res = self.gateway.send(env)
        rec.gateway = {"name": getattr(self.gateway, "name", "?"),
                       "can_move_robot": bool(getattr(self.gateway, "can_move_robot", False)),
                       "sent": bool(res.get("sent")), "shadow": bool(res.get("shadow")),
                       "ipoc": res.get("ipoc")}

        # --- FEEDBACK + REPLAN ------------------------------------------------
        fb = None
        if hasattr(self.gateway, "feedback"):
            fb = self.gateway.feedback()
            if fb and fb.get("ok"):
                fb = {**fb, "commanded": list(one_step[0][:ARM_DIM])}
        rec.feedback = fb
        post = evaluate_stop_signals(configured=conds, observation=observation,
                                     feedback=fb, tolerance_deg=tol)
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
