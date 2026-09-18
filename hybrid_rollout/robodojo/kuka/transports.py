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
RSI_CYCLE_TIME = 0.004            # udp_teleoperate.py:127
RSI_HZ = 1.0 / RSI_CYCLE_TIME     # 250 Hz
RSI_LOCAL_IP_ENV = "KUKA_LOCAL_IP"
RSI_LOCAL_IP_DEFAULT = "172.17.255.2"
RSI_IPOC_FIELD = "IPOC"
# The controller's <Sen> Type attribute is "ImFree", NOT "KUKA"
# (udp_teleoperate.py:283). Verified against the deployed stack.
RSI_RESPONSE_ROOT = "Sen"
RSI_SEN_TYPE = "ImFree"
RSI_LINE_ENDING = "\r\n"
RSI_JOINT_PRECISION = 2           # the stack formats AK values as :.2f

# Gripper does NOT necessarily travel over RSI. The deployed stack offers two
# paths (udp_teleoperate.py:24-34): <GRIPPER_POS> inside the RSI frame via the
# KUKA SPS, or direct Modbus RTU over USB-RS485 from the Jetson. Which one is in
# use is a deployment fact this package does not assume.
GRIPPER_PATHS = ("rsi_gripper_pos", "direct_modbus_rtu")


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


class LocalPi05ProposalSource:
    """pi0.5 proposals from a LOCAL model the operator already runs.

    Replaces the HTTP serving layer that used to live here. The deployment
    engineer has the checkpoint on their own machine, so this package should not
    prescribe how it is served -- it takes a callable, or pre-computed chunks
    from a file, and stays out of the way.

    `infer` receives the observation dict and must return a list of 50 rows of 7
    absolute joint targets in degrees. Anything else is refused.

    THE CHECKPOINT CONTRACT IS STILL ENFORCED. `meta` must report
    use_relative_actions=True. The eight finetunes under oss://i-robot-data/models/
    look correct in every other respect and are a different training run; loading
    one does not crash, it silently returns targets in the wrong space.
    """

    name = "pi05_local"
    is_live = True
    provenance = "model_predicted"

    def __init__(self, infer=None, *, chunks: dict[str, list[list[float]]] | None = None,
                 checkpoint_id: str | None = None,
                 meta: dict[str, Any] | None = None,
                 expected_steps: int = 50, expected_dims: int = 7) -> None:
        if infer is None and chunks is None:
            raise ValueError("supply either infer= (a callable) or chunks= (precomputed)")
        self.infer = infer
        self.chunks = dict(chunks or {})
        self.checkpoint_id = checkpoint_id
        self.meta = dict(meta or {})
        self.expected_steps = expected_steps
        self.expected_dims = expected_dims

    @classmethod
    def from_file(cls, path: str, **kw) -> "LocalPi05ProposalSource":
        """Chunks the operator produced offline: {observation_id: 50x7}."""
        import json as _json
        from pathlib import Path as _P
        raw = _json.loads(_P(path).read_text())
        meta = raw.pop("_meta", {}) if isinstance(raw, dict) else {}
        return cls(chunks={k: v for k, v in raw.items() if not k.startswith("_")},
                   meta=meta, **kw)

    def _check_contract(self) -> str | None:
        if self.meta and self.meta.get("use_relative_actions") is not True:
            return (f"checkpoint reports use_relative_actions="
                    f"{self.meta.get('use_relative_actions')}; expected True. "
                    f"This is almost certainly a different training run -- refusing.")
        return None

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        bad = self._check_contract()
        if bad:
            return {"ok": False, "error": bad}
        oid = observation.get("observation_id")
        try:
            rows = (self.chunks.get(oid) if self.infer is None
                    else self.infer(observation))
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if rows is None:
            return {"ok": False, "error": f"no proposal for {oid}"}
        rows = [list(r) for r in rows]
        if len(rows) != self.expected_steps:
            return {"ok": False,
                    "error": f"expected {self.expected_steps} steps, got {len(rows)}"}
        if any(len(r) != self.expected_dims for r in rows):
            return {"ok": False,
                    "error": f"every row must have {self.expected_dims} values"}
        return {"ok": True, "rows": rows, "checkpoint_id": self.checkpoint_id,
                "source": self.name, "is_live": True,
                "provenance": self.provenance, "meta": self.meta}


class AstraReviewSource:
    """LIVE Astra review. MAKES A PAID API CALL.

    Off unless explicitly constructed with enabled=True and a credential env var
    that is actually populated. `dry_run` builds and hashes the request body and
    returns it WITHOUT sending, so a run can be costed and inspected first.

    The credential is read from the environment by NAME and never stored,
    logged, printed or placed in any returned structure.
    """

    name = "astra_live"
    is_live = True

    def __init__(self, *, base_url: str, model: str, api_key_env: str,
                 enabled: bool = False, dry_run: bool = True,
                 reasoning: str | None = None, store: bool = False,
                 timeout: float = 180.0, transport=None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.enabled = bool(enabled)
        self.dry_run = bool(dry_run)
        self.reasoning = reasoning
        self.store = store
        self.timeout = timeout
        self.transport = transport

    def preflight(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "AstraReviewSource constructed with enabled=False"
        if not os.environ.get(self.api_key_env):
            return False, f"${self.api_key_env} is empty"
        return True, "ready"

    def build_body(self, packet: dict[str, Any],
                   images: list[str] | None = None) -> dict[str, Any]:
        content: list[Any] = [{"type": "input_text", "text": packet["user_text"]}]
        for url in images or []:
            content.append({"type": "input_image", "image_url": url})
        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "system", "content": packet["system"]},
                      {"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "kuka_action_review",
                                "schema": packet["response_schema"], "strict": True}},
            "store": bool(self.store)}
        if self.reasoning:
            body["reasoning"] = {"effort": self.reasoning}
        return body

    def review(self, packet: dict[str, Any]) -> dict[str, Any]:
        import hashlib as _h
        import json as _json
        body = self.build_body(packet, packet.get("image_data_urls"))
        digest = _h.sha256(_json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        if self.dry_run:
            return {"ok": False, "dry_run": True, "body_sha256_12": digest,
                    "would_send_bytes": len(_json.dumps(body)),
                    "model": self.model, "endpoint": self.base_url,
                    "note": "DRY RUN -- nothing sent. Set dry_run=False to call."}
        ok, why = self.preflight()
        if not ok:
            return {"ok": False, "error": why, "body_sha256_12": digest}
        try:
            raw = (self.transport or _urllib_post)(
                self.base_url, body,
                {"Authorization": f"Bearer {os.environ[self.api_key_env]}",
                 "Content-Type": "application/json"}, self.timeout)
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                    "body_sha256_12": digest}
        text, usage = _extract_text(raw)
        if text is None:
            return {"ok": False, "error": "no text in response", "usage": usage,
                    "body_sha256_12": digest}
        try:
            decision = _json.loads(text)
        except Exception:
            return {"ok": False, "error": "response was not valid JSON",
                    "raw_text": text[:2000], "body_sha256_12": digest}
        return {"ok": True, "decision": decision, "usage": usage,
                "source": self.name, "is_live": True, "body_sha256_12": digest}


def _urllib_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """One attempt. No retries, no fallback model."""
    import json as _json
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=_json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:1500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from None


def _extract_text(raw: Any) -> tuple[str | None, dict | None]:
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(raw, dict):
        return None, usage
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"], usage
    for item in raw.get("output") or []:
        for part in (item or {}).get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"], usage
    return None, usage


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


def rsi_response_xml(ipoc: int, joints_deg: Sequence[float], *,
                     gripper_pos: int = 0, stop_flag: int = 0) -> str:
    """The <Sen> frame the Jetson returns to the controller, IPOC echoed.

    Byte-compatible with the deployed builder in
    KUKA/teleoperation/udp_teleoperate.py:282-291 -- same Type, same CRLF line
    endings, same 2-decimal AK formatting, and the same GRIPPER_POS/Stopflag
    elements. An earlier version of this function was wrong on all four counts;
    it is now pinned by a test so it cannot drift from the real stack again.

    NOTHING IN THIS PACKAGE SENDS IT. This exists so the deployment engineer can
    diff it against the real gateway.
    """
    j = list(joints_deg[:6])
    nl = "\r\n"
    ak = " ".join(f'A{i + 1}="{v:.{RSI_JOINT_PRECISION}f}"' for i, v in enumerate(j))
    return (f'<{RSI_RESPONSE_ROOT} Type="{RSI_SEN_TYPE}">{nl}'
            f'<AK {ak}/>{nl}'
            f'<GRIPPER_POS>{gripper_pos}</GRIPPER_POS>{nl}'
            f'<Stopflag>{stop_flag}</Stopflag>{nl}'
            f'<{RSI_IPOC_FIELD}>{ipoc}</{RSI_IPOC_FIELD}>{nl}'
            f'</{RSI_RESPONSE_ROOT}>')


def gripper_to_raw(normalised: float) -> int:
    """0..1 chunk value -> the raw integer the deployed stack sends."""
    from .contract import GRIPPER_SCALE
    return int(min(1.0, max(0.0, float(normalised))) * GRIPPER_SCALE)
