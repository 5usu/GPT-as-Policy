"""Arming and gating for YAM, on the KUKA branch's safety primitives.

What is REUSED unchanged from `kuka.safety`: the four modes and the structural
rule that replay/live_shadow cannot build a command; the supervisor (heartbeat,
deadman, E-stop) check; observation freshness; the signed single-use envelope;
the replay-refusing ledger; the signing key read from the environment by name.

What is YAM-SPECIFIC: the list of values that must be measured on this rig
before anything may arm, and the identity of the rig. The KUKA list asks for a
dishwasher handle pose and a KUKA $TOOL; none of that describes two YAM arms.

THE CENTRAL RULE IS UNCHANGED: NEVER INVENT A NUMBER. Every entry in
REQUIRED_CONFIG is supplied by a human who measured or decided it on this rig.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..kuka.safety import (COMMAND_CAPABLE_MODES, ArmingRefused, CommandEnvelope,
                           CommandLedger, Mode, Supervisor, check_freshness,
                           require_command_capable, signing_key_from_env)

SCHEMA = "hybrid_rollout.robodojo.yam.safety.v1"

__all__ = ["ASTRA_DIRECT_EXTRA", "COMMAND_CAPABLE_MODES", "ArmingRefused",
           "CommandEnvelope", "CommandLedger", "Mode", "REQUIRED_CONFIG",
           "RigIdentity", "Supervisor", "authorise", "check_allowlist",
           "identity_from_config", "missing_config", "require_command_capable",
           "signing_key_from_env"]

REQUIRED_CONFIG: dict[str, str] = {
    # --- rig -----------------------------------------------------------------
    "camera_mapping": ("which /dev/video node is top, left and right, confirmed "
                       "by LOOKING at each stream; a swapped wrist pair gives "
                       "confident actions for a scene the policy is not seeing"),
    "gripper_limits_left": "calibrated [closed, open] motor limits, left gripper",
    "gripper_limits_right": "calibrated [closed, open] motor limits, right gripper",
    "rest_pose": ("the pose both arms can safely go limp in; the motors are "
                  "unpowered when the process exits"),
    "table_workspace": "the region both arms may occupy for this task",
    "estop_tested": "reference to a completed test of the hardware E-stop on this rig",
    # --- limits --------------------------------------------------------------
    "max_speed": "joint speed ceiling for this task, rad/s",
    "max_acceleration": "joint acceleration ceiling, rad/s^2",
    "max_step_displacement": "largest displacement for one supervised step, rad",
    # --- supervision ---------------------------------------------------------
    "observation_freshness_s": "maximum observation age still considered current",
    "heartbeat_timeout_s": "supervisor heartbeat timeout",
    "success_predicate": "measurable predicate defining task success",
    "commanded_observed_tolerance_rad": ("allowed commanded-vs-measured drift; "
                                         "unset makes that stop condition "
                                         "unevaluable, which stops the run"),
}

ASTRA_DIRECT_EXTRA: dict[str, str] = {
    "astra_direct_authorised_by": "named human who authorised direct control",
    "astra_direct_bounded_action_space": "explicit bounded action space Astra may propose within",
    "astra_direct_dry_run_passed": "reference to a completed shadow campaign on this task",
    "collision_model": "inter-arm and table collision model",
}


def missing_config(config: dict[str, Any], mode: Mode) -> list[str]:
    """Names of required entries that are absent or null. Never guesses."""
    required = dict(REQUIRED_CONFIG)
    if mode is Mode.ASTRA_DIRECT:
        required.update(ASTRA_DIRECT_EXTRA)
    return sorted(k for k in required if config.get(k) in (None, "", [], {}))


@dataclass(frozen=True)
class RigIdentity:
    """The exact rig this build may address: both CAN channels, by name."""
    robot_model: str
    rig_id: str
    left_can: str
    right_can: str

    def key(self) -> str:
        return f"{self.robot_model}|{self.rig_id}|{self.left_can}+{self.right_can}"


IDENTITY_FIELDS = ("robot_model", "rig_id", "left_can", "right_can")


def identity_from_config(cfg: dict) -> RigIdentity:
    missing = [f for f in IDENTITY_FIELDS if not cfg.get(f)]
    if missing:
        raise ArmingRefused(
            "identity_incomplete",
            "the rig this build may address is not fully stated; missing "
            + ", ".join(missing) + ". These are operator facts and are never "
            "defaulted.")
    return RigIdentity(*(str(cfg[f]) for f in IDENTITY_FIELDS))


def check_allowlist(target: RigIdentity, allowlist: Iterable[RigIdentity]) -> None:
    keys = {a.key() for a in allowlist}
    if target.key() not in keys:
        raise ArmingRefused("not_allowlisted",
                            f"{target.key()} is not in the {len(keys)}-entry allowlist")


def authorise(*, mode: Mode, config: dict[str, Any], target: RigIdentity,
              allowlist: Iterable[RigIdentity], supervisor: Supervisor,
              ledger: CommandLedger, command_id: str,
              rows: Sequence[Sequence[float]], control_hz: float,
              observation_id: str, observation_epoch: float | None,
              decision_mode: str, audit: dict[str, Any],
              secret: bytes | None, now: float | None = None,
              execution_safe: bool = False) -> CommandEnvelope:
    """The single gate, in the KUKA order. Every check must pass."""
    now = time.time() if now is None else now
    require_command_capable(mode)                                   # 1 structural
    miss = missing_config(config, mode)                             # 2 config
    if miss:
        raise ArmingRefused("missing_config",
                            f"{len(miss)} required value(s) not supplied; refusing "
                            f"to invent them", missing=miss)
    check_allowlist(target, allowlist)                              # 3 identity
    supervisor.check(now=now, timeout_s=float(config["heartbeat_timeout_s"]))  # 4
    age = check_freshness(observation_epoch, now=now,               # 5 freshness
                          max_age_s=float(config["observation_freshness_s"]))
    if not execution_safe:                                          # 6 validation
        raise ArmingRefused("not_execution_safe",
                            "candidate failed deterministic YAM validation")
    rows = [list(r) for r in rows]
    if len(rows) != 1:                                              # 7 single step
        raise ArmingRefused("not_single_step",
                            f"supervised execution emits exactly 1 step, got {len(rows)}")
    if not secret:                                                  # 8 signing
        raise ArmingRefused("no_signing_key", "no envelope signing key configured")
    env = CommandEnvelope(
        command_id=command_id, mode=mode.value, robot=target.key(), rows=rows,
        n_steps=len(rows), control_hz=control_hz, issued_at=now,
        observation_id=observation_id, observation_age_s=round(age, 4),
        decision_mode=decision_mode, audit=dict(audit))
    ledger.reserve(command_id, env.digest())                        # 9 one-time
    env.sign(secret)
    env.approved_for_execution = True
    return env
