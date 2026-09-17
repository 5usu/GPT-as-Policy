"""Boundaries: A800 (proposals), Astra (review), Jetson (the only command path).

ROLES, AND WHY THEY ARE SPLIT

  eng-1   development and offline analysis. Never talks to the robot. Holds the
          current OPENAI_API_KEY, which is NOT to be copied anywhere.
  A800    GPU box. Runs pi0.5 inference and returns proposed action chunks.
          PROPOSAL-ONLY BY DEFAULT. It has no route to the robot and no command
          code path. Live Astra review from the A800 requires BOTH an explicit
          A800_LIVE_REVIEW=1 and a NEW dedicated key supplied later.
  Jetson  the ONLY KUKA command gateway. Physically on the robot network, speaks
          RSI UDP/XML at 250 Hz. Everything that could move the arm lives here.

Nothing in this file performs I/O on import, and the default gateway is a shadow
gateway that records what it would have sent and sends nothing.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .safety import ArmingRefused, CommandEnvelope

SCHEMA = "hybrid_rollout.robodojo.kuka.transports.v1"

# Evidence-backed from the existing teleoperation stack (udp_teleoperate.py):
# the Jetson binds a local UDP port and the KUKA controller connects to it; the
# controller sends <Rob> with AIPos/RIst and expects <Sen> back carrying the same
# IPOC. That IPOC echo is what makes an RSI exchange idempotent.
RSI_DEFAULT_PORT = 59152
RSI_HZ = 250.0
RSI_REQUEST_ROOT = "Rob"
RSI_RESPONSE_ROOT = "Sen"
RSI_IPOC_FIELD = "IPOC"


class ProposalSource(Protocol):
    name: str
    is_live: bool

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]: ...


class ReviewSource(Protocol):
    name: str
    is_live: bool

    def review(self, packet: dict[str, Any]) -> dict[str, Any]: ...


class CommandGateway(Protocol):
    name: str
    can_move_robot: bool

    def send(self, envelope: CommandEnvelope) -> dict[str, Any]: ...


# ------------------------------------------------------------------- A800 side
def a800_live_review_enabled() -> tuple[bool, str]:
    """Live Astra from the A800 is off unless explicitly switched on.

    Two independent things are required and neither implies the other: the flag
    says a human intends live review, the key says one exists. The key must be a
    NEW dedicated credential -- the eng-1 key is not to be copied to the A800.
    """
    flag = os.environ.get("A800_LIVE_REVIEW") == "1"
    key = os.environ.get("A800_ASTRA_API_KEY")
    if not flag:
        return False, "A800_LIVE_REVIEW is not set to 1; A800 stays proposal-only"
    if not key:
        return False, ("A800_LIVE_REVIEW=1 but A800_ASTRA_API_KEY is empty. Supply "
                       "a NEW dedicated key on the A800; do not copy the eng-1 "
                       "OPENAI_API_KEY")
    return True, "live review enabled by explicit flag and dedicated key"


class OfflineProposalSource:
    """Replays pi0.5 chunks recorded earlier. No GPU, no model, no network."""

    name = "recorded_pi05"
    is_live = False

    def __init__(self, chunks_by_observation: dict[str, list[list[float]]],
                 checkpoint_id: str | None = None) -> None:
        self.chunks = dict(chunks_by_observation)
        self.checkpoint_id = checkpoint_id

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        oid = observation.get("observation_id")
        rows = self.chunks.get(oid)
        if rows is None:
            return {"ok": False, "error": f"no recorded proposal for {oid}"}
        return {"ok": True, "rows": [list(r) for r in rows],
                "checkpoint_id": self.checkpoint_id, "source": self.name,
                "is_live": False}


class RecordedTrajectorySource:
    """Phase 2's proposal source: the RECORDED HUMAN DEMONSTRATION.

    This is deliberately not pi0.5. Phase 2 asks what a successful episode looks
    like by actually performing one, and the safest trajectory to perform is the
    one a human already performed successfully on this cell. It also has a
    property the model output does not: measured on real val_ood data the demo's
    largest target-to-target step is 3.089 deg, comfortably inside joint_2's
    6.667 deg/step cap at 30 Hz, whereas the pi0.5 chunk's 6.992 deg A2 step
    exceeds it. The demo is executable as recorded; the model proposal is not.

    Provenance is reported as `recorded_demo` and must stay that way: a
    demonstration replayed through the gates is evidence about the task, never a
    model prediction.
    """

    name = "recorded_demo_trajectory"
    is_live = False
    provenance = "recorded_demo"

    def __init__(self, chunks_by_observation: dict[str, list[list[float]]],
                 episode_id: str | None = None) -> None:
        self.chunks = dict(chunks_by_observation)
        self.episode_id = episode_id

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        oid = observation.get("observation_id")
        rows = self.chunks.get(oid)
        if rows is None:
            return {"ok": False, "error": f"no recorded trajectory at {oid}"}
        return {"ok": True, "rows": [list(r) for r in rows],
                "checkpoint_id": None, "source": self.name, "is_live": False,
                "provenance": self.provenance,
                "note": ("recorded human demonstration; NOT a model proposal")}


class BoundedAstraProposalSource:
    """Phase 3: Astra proposes its own action inside an explicit bounded space.

    Refuses to produce anything unless a bounded action space was configured.
    There is no default bound -- an unbounded self-proposal is the one thing this
    whole package exists to prevent.
    """

    name = "astra_direct"
    is_live = False
    provenance = "astra_direct"

    def __init__(self, proposer=None, bounded_action_space: Any = None) -> None:
        self.proposer = proposer
        self.bounds = bounded_action_space

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.bounds in (None, "", [], {}):
            return {"ok": False,
                    "error": ("no bounded_action_space configured; direct "
                              "proposal refused")}
        if self.proposer is None:
            return {"ok": False, "error": "no direct proposer configured"}
        rows = self.proposer(observation, self.bounds)
        if not rows:
            return {"ok": False, "error": "proposer returned nothing"}
        return {"ok": True, "rows": [list(r) for r in rows], "source": self.name,
                "is_live": False, "provenance": self.provenance}


class UnconfiguredReview:
    """Default review source. Fails loudly rather than pretending."""

    name = "unconfigured"
    is_live = False

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "error": "no review source configured"}


class RecordedReview:
    """Replays Astra decisions already obtained. Makes no API call, ever."""

    name = "recorded_astra"
    is_live = False

    def __init__(self, decisions: dict[str, dict[str, Any]]) -> None:
        self.decisions = dict(decisions)

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        d = self.decisions.get(packet.get("request_id"))
        if d is None:
            return {"ok": False, "error": "no recorded decision"}
        return {"ok": True, "decision": dict(d), "source": self.name,
                "is_live": False}


# ----------------------------------------------------------------- Jetson side
@dataclass
class ShadowGateway:
    """Default gateway. Records what it WOULD emit and emits nothing.

    can_move_robot is False and there is no socket in this class at all -- shadow
    mode is the absence of a transport, not a disabled one.
    """
    name: str = "jetson_shadow"
    can_move_robot: bool = False
    emitted: list[dict[str, Any]] = field(default_factory=list)

    def send(self, envelope: CommandEnvelope) -> dict[str, Any]:
        record = {"would_send": envelope.to_log(),
                  "rsi": {"port": RSI_DEFAULT_PORT, "hz": RSI_HZ,
                          "response_root": RSI_RESPONSE_ROOT,
                          "ipoc_echo_required": True},
                  "sent": False,
                  "note": ("SHADOW: no UDP socket exists in this gateway. The "
                           "envelope was signed and validated, then logged.")}
        self.emitted.append(record)
        return {"ok": True, "shadow": True, "sent": False, "record": record}


@dataclass
class FakeKukaGateway:
    """Test double for the Jetson RSI gateway. Never touches a network.

    Exists so the gates can be tested end to end. It counts what it receives so
    a test can prove exactly one idempotent command arrives after all gates pass,
    and it refuses an unsigned or replayed envelope the way the real one must.
    """
    secret: bytes
    name: str = "fake_kuka"
    can_move_robot: bool = True
    received: list[dict[str, Any]] = field(default_factory=list)
    _ipoc: int = 0

    def send(self, envelope: CommandEnvelope) -> dict[str, Any]:
        if not envelope.approved_for_execution:
            raise ArmingRefused("unapproved_envelope",
                                "envelope was never authorised")
        if not envelope.verify(self.secret):
            raise ArmingRefused("bad_signature", "envelope signature invalid")
        digest = envelope.digest()
        if any(r["envelope_sha256"] == digest for r in self.received):
            raise ArmingRefused("duplicate_envelope",
                                f"envelope {digest[:12]} already delivered")
        self._ipoc += 1
        rec = {"envelope_sha256": digest, "command_id": envelope.command_id,
               "n_steps": envelope.n_steps, "ipoc": self._ipoc,
               "rows": [list(r) for r in envelope.rows]}
        self.received.append(rec)
        return {"ok": True, "sent": True, "ipoc": self._ipoc,
                "response_root": RSI_RESPONSE_ROOT, "record": rec}

    def feedback(self, achieved: Sequence[float] | None = None) -> dict[str, Any]:
        """Measured state after the step, as RSI would report it."""
        if not self.received:
            return {"ok": False, "error": "nothing executed"}
        rows = self.received[-1]["rows"][0]
        return {"ok": True, "ipoc": self._ipoc,
                "measured": list(achieved) if achieved is not None else list(rows)}


def rsi_response_xml(ipoc: int, joints_deg: Sequence[float]) -> str:
    """The <Sen> frame the Jetson returns to the controller, IPOC echoed.

    Included for the deployment engineer to check against the real .src files in
    teleoperation/trigger_RSI. It is NOT sent by anything in this package.
    """
    ak = " ".join(f'A{i + 1}="{v:.4f}"' for i, v in enumerate(joints_deg[:6]))
    return (f'<{RSI_RESPONSE_ROOT} Type="KUKA">'
            f'<AK {ak}/>'
            f'<{RSI_IPOC_FIELD}>{ipoc}</{RSI_IPOC_FIELD}>'
            f'</{RSI_RESPONSE_ROOT}>')
