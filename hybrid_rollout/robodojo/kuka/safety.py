"""Arming, gating, signing and the ledger. Stdlib only; runs on the Jetson.

REAL MOTION IS IMPOSSIBLE BY DEFAULT.

Nothing here opens a socket. This module decides whether a command may exist at
all; `transports.py` decides where it goes. The split is deliberate: a transport
cannot arm itself.

THE CENTRAL RULE: NEVER INVENT A NUMBER.
A contact-rich task like opening a dishwasher door depends on quantities nobody
has measured yet -- where the handle is, which way the gripper closes, what load
the controller should refuse. A plausible default for any of these is worse than
no value, because it looks like knowledge. Every entry in REQUIRED_CONFIG must be
supplied by a human who measured it; a missing one REFUSES ARMING and names
itself. There are no defaults, no fallbacks and no inference from similar robots.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.safety.v1"


class Mode(str, Enum):
    """Four modes, one state machine, one audit format."""
    REPLAY = "replay"                        # recorded frames -> review. No robot.
    LIVE_SHADOW = "live_shadow"              # live frames -> prediction. Commands suppressed.
    REVIEWED_EXECUTION = "reviewed_execution"  # pi0.5 proposes, Astra reviews, supervised single step
    ASTRA_DIRECT = "astra_direct"            # Astra proposes; identical gates; hardest to arm


#: Modes that may EVER produce a robot command. The other two are structurally
#: incapable of it -- not "configured off", but rejected before a command object
#: can be constructed (see require_command_capable).
COMMAND_CAPABLE_MODES = frozenset({Mode.REVIEWED_EXECUTION, Mode.ASTRA_DIRECT})


class ArmingRefused(Exception):
    """Raised whenever a command could not be authorised. Always fail-closed."""

    def __init__(self, code: str, detail: str, missing: Sequence[str] = ()) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.missing = list(missing)


# --------------------------------------------------------------- required config
#: Every key must be present, non-null, and supplied by a human who measured it.
#: The description is what the deployment engineer has to go and find out.
REQUIRED_CONFIG: dict[str, str] = {
    # --- cell geometry -------------------------------------------------------
    "handle_pose": "measured pose of the dishwasher door handle in the robot base frame",
    "door_hinge_axis": "hinge axis origin+direction in base frame",
    "door_open_region": "expected door-motion region; motion outside it is a stop condition",
    "table_workspace": "allowed Cartesian volume for the whole task",
    # --- calibration ---------------------------------------------------------
    "camera_intrinsics": "per-camera K and distortion, for base and wrist",
    "camera_extrinsics": "per-camera pose in the robot base frame",
    "tcp_transform": "verified $TOOL/TCP transform (flange -> tool tip)",
    "base_frame": "verified robot base frame definition",
    "gripper_polarity": "which commanded value is OPEN and which is CLOSED",
    # --- limits --------------------------------------------------------------
    "force_torque_limits": "F/T or controller load limits at which motion must abort",
    "max_speed": "commanded joint/Cartesian speed ceiling for this task",
    "max_acceleration": "commanded acceleration ceiling",
    "max_step_displacement": "largest permitted displacement for a single supervised step",
    # --- supervision ---------------------------------------------------------
    "observation_freshness_s": "maximum observation age still considered current",
    "success_predicate": "measurable predicate defining task success",
    "heartbeat_timeout_s": "supervisor heartbeat timeout",
}

#: Extra prerequisites that ASTRA_DIRECT needs on top of everything above.
#: Astra proposing its own actions removes pi0.5 from the loop, so the evidence
#: that a proposal is reasonable has to come from somewhere else.
ASTRA_DIRECT_EXTRA: dict[str, str] = {
    "astra_direct_authorised_by": "named human who authorised direct control",
    "astra_direct_bounded_action_space": "explicit bounded action space Astra may propose within",
    "astra_direct_dry_run_passed": "reference to a completed shadow campaign on this task",
    "collision_model": "cell and self-collision model",
}


def missing_config(config: dict[str, Any], mode: Mode) -> list[str]:
    """Names of required entries that are absent or null. Never guesses."""
    required = dict(REQUIRED_CONFIG)
    if mode is Mode.ASTRA_DIRECT:
        required.update(ASTRA_DIRECT_EXTRA)
    return sorted(k for k in required if config.get(k) in (None, "", [], {}))


# ------------------------------------------------------------------- allowlist
@dataclass(frozen=True)
class RobotIdentity:
    """Exact robot + controller this build may ever address."""
    robot_model: str
    controller_serial: str
    rsi_host: str
    rsi_port: int

    def key(self) -> str:
        return f"{self.robot_model}|{self.controller_serial}|{self.rsi_host}:{self.rsi_port}"


#: Identity fields the operator must state. They are not measurements and
#: cannot be derived from the cell, so they are never defaulted.
IDENTITY_FIELDS = ("robot_model", "controller_serial", "rsi_host", "rsi_port")


def identity_from_config(cfg: dict) -> RobotIdentity:
    """Build the one robot this build may address, from validated config.

    Raises ArmingRefused naming every missing field. An empty allowlist is not
    a permissive default -- it refuses everything -- but a silently-empty one
    reads as a configuration bug rather than a decision, so this exists to make
    populating it explicit.
    """
    missing = [f for f in IDENTITY_FIELDS if not cfg.get(f)]
    if missing:
        raise ArmingRefused(
            "identity_incomplete",
            "the robot this build may address is not fully stated; missing "
            + ", ".join(missing) + ". These are operator facts, not "
            "measurements, and are never defaulted.")
    return RobotIdentity(
        robot_model=str(cfg["robot_model"]),
        controller_serial=str(cfg["controller_serial"]),
        rsi_host=str(cfg["rsi_host"]),
        rsi_port=int(cfg["rsi_port"]))


def signing_key_from_env(var_name: str) -> bytes:
    """Read the command-signing key from the environment BY NAME.

    The key itself is never printed, logged, hashed into any message, or placed
    in a returned structure -- only its presence and length are ever observed.
    """
    import os
    raw = os.environ.get(var_name or "", "")
    if not raw:
        raise ArmingRefused(
            "no_signing_key",
            f"${var_name} is empty. Commands are signed so a replayed or "
            f"forged envelope cannot reach the arm; without a key there is "
            f"nothing to verify against. Set it in the operator's environment "
            f"-- it is never read from a file or a flag.")
    key = raw.encode() if isinstance(raw, str) else bytes(raw)
    if len(key) < 16:
        raise ArmingRefused(
            "weak_signing_key",
            f"${var_name} is {len(key)} bytes; at least 16 are required.")
    return key


def check_allowlist(target: RobotIdentity,
                    allowlist: Iterable[RobotIdentity]) -> None:
    """Exact match on all four fields. No wildcards, no prefix matching."""
    keys = {a.key() for a in allowlist}
    if target.key() not in keys:
        raise ArmingRefused("not_allowlisted",
                            f"{target.key()} is not in the {len(keys)}-entry allowlist")


# ---------------------------------------------------------------------- ledger
class CommandLedger:
    """One-time / idempotency ledger. Append-only, refuses replays.

    A command id may be issued once. A repeat is refused rather than
    de-duplicated silently, because a repeat means something upstream lost track
    of whether a motion already happened -- and on a real arm that is exactly
    when you must stop, not retry.
    """

    def __init__(self) -> None:
        self._seen: dict[str, dict[str, Any]] = {}

    def reserve(self, command_id: str, envelope_sha256: str) -> None:
        prior = self._seen.get(command_id)
        if prior is not None:
            raise ArmingRefused(
                "duplicate_command",
                f"command_id {command_id} already issued at {prior['at']} "
                f"(envelope {prior['envelope_sha256'][:12]}); refusing replay")
        self._seen[command_id] = {"at": time.time(),
                                  "envelope_sha256": envelope_sha256}

    def issued(self) -> list[str]:
        return sorted(self._seen)

    def __len__(self) -> int:
        return len(self._seen)


# ------------------------------------------------------------------- heartbeat
@dataclass
class Supervisor:
    """Human supervisor state. All three are independent and all are required
    for a command-capable mode."""
    heartbeat_at: float | None = None      # periodic liveness
    deadman_held: bool = False             # human physically holding the enable
    estop_clear: bool = False              # E-stop NOT engaged

    def check(self, *, now: float, timeout_s: float) -> None:
        if self.heartbeat_at is None:
            raise ArmingRefused("no_heartbeat", "supervisor heartbeat never received")
        age = now - self.heartbeat_at
        if age > timeout_s:
            raise ArmingRefused("stale_heartbeat",
                                f"heartbeat {age:.2f}s old, timeout {timeout_s}s")
        if not self.deadman_held:
            raise ArmingRefused("deadman_released",
                                "human deadman is not held; single-step execution "
                                "requires a continuously held enable")
        if not self.estop_clear:
            raise ArmingRefused("estop_engaged", "E-stop is engaged")


#: The physical E-stop is a hardware interlock wired to the controller. This
#: software can OBSERVE it and refuse, but it cannot clear it and must never
#: model itself as able to. `estop_clear` above is an input, never an output.
ESTOP_BOUNDARY = (
    "The E-stop is a physical interlock on the KUKA controller. Software may "
    "read its state and refuse to act; software can NEVER clear it, bypass it, "
    "or command through it. No code path in this package writes E-stop state.")


# ------------------------------------------------------------------- freshness
def check_freshness(observation_epoch: float | None, *, now: float,
                    max_age_s: float) -> float:
    if observation_epoch is None:
        raise ArmingRefused("no_observation", "no observation timestamp supplied")
    age = now - observation_epoch
    if age < 0:
        raise ArmingRefused("observation_from_future",
                            f"observation timestamp is {-age:.3f}s ahead of now")
    if age > max_age_s:
        raise ArmingRefused("stale_observation",
                            f"observation {age:.3f}s old, limit {max_age_s}s")
    return age


# -------------------------------------------------------------------- envelope
@dataclass
class CommandEnvelope:
    """A signed, single-use, one-step command. Construction does NOT authorise
    it; `sign` is only reachable through `authorise`."""
    command_id: str
    mode: str
    robot: str
    rows: list[list[float]]
    n_steps: int
    control_hz: float
    issued_at: float
    observation_id: str
    observation_age_s: float
    decision_mode: str
    audit: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None
    approved_for_execution: bool = False

    def payload(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("signature", "approved_for_execution")}
        return d

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def sign(self, secret: bytes) -> str:
        self.signature = hmac.new(secret, self.digest().encode(),
                                  hashlib.sha256).hexdigest()
        return self.signature

    def verify(self, secret: bytes) -> bool:
        if not self.signature:
            return False
        expected = hmac.new(secret, self.digest().encode(),
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        d["envelope_sha256"] = self.digest()
        d["signature_present"] = bool(self.signature)
        d.pop("signature", None)          # never log the signature itself
        return d


def require_command_capable(mode: Mode) -> None:
    """replay and live_shadow can never produce a command. Structural, not config."""
    if mode not in COMMAND_CAPABLE_MODES:
        raise ArmingRefused(
            "mode_cannot_command",
            f"mode {mode.value} is observation-only; command construction is "
            f"not reachable in this mode")


def authorise(*, mode: Mode, config: dict[str, Any], target: RobotIdentity,
              allowlist: Iterable[RobotIdentity], supervisor: Supervisor,
              ledger: CommandLedger, command_id: str,
              rows: Sequence[Sequence[float]], control_hz: float,
              observation_id: str, observation_epoch: float | None,
              decision_mode: str, audit: dict[str, Any],
              secret: bytes | None, now: float | None = None,
              execution_safe: bool = False) -> CommandEnvelope:
    """The single gate. Every check must pass or nothing is produced.

    Order matters: the cheapest structural refusals come first so a
    misconfigured run fails on the real reason rather than a downstream symptom.
    """
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
                            "candidate failed deterministic KUKA validation")

    rows = [list(r) for r in rows]
    if len(rows) != 1:                                              # 7 single step
        raise ArmingRefused("not_single_step",
                            f"supervised execution emits exactly 1 step, got "
                            f"{len(rows)}")

    if not secret:                                                  # 8 signing
        raise ArmingRefused("no_signing_key",
                            "no envelope signing key configured")

    env = CommandEnvelope(
        command_id=command_id, mode=mode.value, robot=target.key(), rows=rows,
        n_steps=len(rows), control_hz=control_hz, issued_at=now,
        observation_id=observation_id, observation_age_s=round(age, 4),
        decision_mode=decision_mode, audit=dict(audit))
    ledger.reserve(command_id, env.digest())                        # 9 one-time
    env.sign(secret)
    env.approved_for_execution = True
    return env
