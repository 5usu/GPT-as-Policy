"""Boundaries on YAM: the pi0.5 server (proposals), Astra (review), the rig.

REUSED FROM THE KUKA BRANCH, UNCHANGED
  AstraReviewSource   background submit+poll, true wall-clock deadline, one
                      attempt, never re-asks an answered review. Only the
                      json_schema name in the request is renamed.
  RecordedTrajectorySource, RecordedReview, UnconfiguredReview

YAM-SPECIFIC
  the checkpoint contract. The KUKA source refused any chunk whose meta did not
  say use_relative_actions=True; that flag identifies the KUKA fine-tune and
  means nothing here. The YAM check is on what identifies THIS checkpoint: the
  repo, the norm-stats digest, the action layout and the gripper convention.
  A different YAM fine-tune loads fine and returns plausible targets for a
  different normalisation -- the same silent failure the KUKA check existed for.

Nothing here performs I/O on import. The default gateway records and sends
nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..kuka.safety import CommandEnvelope
from ..kuka.transports import (AstraReviewSource, RecordedReview,
                               RecordedTrajectorySource, UnconfiguredReview)
from .contract import ACTION_DIM, CHECKPOINT, CHUNK_STEPS

SCHEMA = "hybrid_rollout.robodojo.yam.transports.v1"

__all__ = ["CONTRACT_KEYS", "Pi05ChunkReplay", "Pi05HttpProposalSource",
           "RecordedReview", "RecordedTrajectorySource", "ShadowGateway",
           "UnconfiguredReview", "YamAstraReviewSource", "check_checkpoint_meta",
           "rows_ok"]

#: What a pi0.5 response must report, and must match, before its chunk is used.
CONTRACT_KEYS = ("repo_id", "norm_stats_sha256", "action_horizon", "action_dim",
                 "action_space", "gripper_convention")


def check_checkpoint_meta(meta: dict[str, Any] | None) -> str | None:
    """None if `meta` describes the contracted checkpoint, else why not."""
    if not meta:
        return ("pi0.5 response carries no checkpoint meta; refusing a chunk "
                "whose producer cannot be identified")
    bad = [f"{k}={meta.get(k)!r} (expected {CHECKPOINT[k]!r})"
           for k in CONTRACT_KEYS if meta.get(k) != CHECKPOINT[k]]
    if bad:
        return ("pi0.5 checkpoint does not match the YAM contract: "
                + "; ".join(bad) + ". A different checkpoint returns plausible "
                "targets in a different normalisation -- refusing.")
    return None


def rows_ok(rows: Any) -> str | None:
    if not isinstance(rows, list) or len(rows) != CHUNK_STEPS:
        n = len(rows) if isinstance(rows, list) else "?"
        return f"expected {CHUNK_STEPS} rows, got {n}"
    for i, r in enumerate(rows):
        if not isinstance(r, list) or len(r) != ACTION_DIM:
            return f"row {i} must have {ACTION_DIM} values"
        for v in r:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v \
                    or v in (float("inf"), float("-inf")):
                return f"row {i} contains {v!r}"
    return None


class Pi05HttpProposalSource:
    """Live proposals from `yam.pi05_serve` over HTTP. Proposal-only.

    `observation` needs `state` (14), `images` {top,left,right: data URL or
    base64 JPEG/PNG}, `task`. The loopback server is reached without the
    outbound proxy, for the reason cli.cmd_live gives on the KUKA branch.
    """

    name = "pi05_http"
    is_live = True
    provenance = "model_predicted"

    def __init__(self, url: str, *, timeout_s: float = 30.0, opener=None) -> None:
        import urllib.request
        self.url = url
        self.timeout_s = float(timeout_s)
        self._open = (opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}))).open
        self.meta: dict[str, Any] = {}

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request
        body = json.dumps({"state": list(observation.get("state") or []),
                           "images": observation.get("images") or {},
                           "task": observation.get("task", ""),
                           "observation_id": observation.get("observation_id")}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with self._open(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            # The server refuses with a reason in the body; keep it.
            try:
                detail = json.loads(exc.read().decode()).get("error")
            except Exception:                                  # noqa: BLE001
                detail = None
            return {"ok": False, "error": f"HTTP {exc.code}: {detail or exc.reason}"[:300]}
        except Exception as exc:                               # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if not out.get("ok"):
            return {"ok": False, "error": str(out.get("error", "server error"))[:300]}
        bad = check_checkpoint_meta(out.get("meta"))
        if bad is None:
            bad = rows_ok(out.get("rows"))
        if bad:
            return {"ok": False, "error": bad}
        self.meta = dict(out["meta"])
        return {"ok": True, "rows": out["rows"], "checkpoint_id": _ckpt_id(self.meta),
                "source": self.name, "is_live": True, "provenance": self.provenance,
                "latency_s": out.get("latency_s"), "meta": self.meta}


class Pi05ChunkReplay:
    """Precomputed chunks from `yam.make_chunks`: {observation_id: 16x14, _meta}.

    `sequential=True` serves them in order, for the live loop, where ids
    cannot match; the audit then says `recorded_chunk_replay`, never live.
    """

    name = "pi05_replay"
    is_live = False
    provenance = "model_predicted"

    def __init__(self, chunks: dict[str, list[list[float]]], meta: dict[str, Any],
                 *, sequential: bool = False) -> None:
        self.chunks = {k: v for k, v in chunks.items() if not str(k).startswith("_")}
        self.meta = dict(meta or {})
        self.sequential = bool(sequential)
        self._order = list(self.chunks)
        self._next = 0

    @classmethod
    def from_file(cls, path: str, **kw) -> "Pi05ChunkReplay":
        from pathlib import Path
        raw = json.loads(Path(path).read_text())
        return cls(raw, raw.get("_meta") or {}, **kw)

    def propose(self, observation: dict[str, Any]) -> dict[str, Any]:
        bad = check_checkpoint_meta(self.meta)
        if bad:
            return {"ok": False, "error": bad}
        oid = observation.get("observation_id")
        if self.sequential:
            if not self._order:
                return {"ok": False, "error": "no chunks to replay"}
            key = self._order[self._next % len(self._order)]
            self._next += 1
        else:
            key = oid
        rows = self.chunks.get(key)
        if rows is None:
            return {"ok": False, "error": f"no proposal for {oid}"}
        bad = rows_ok(rows)
        if bad:
            return {"ok": False, "error": bad}
        out = {"ok": True, "rows": [list(r) for r in rows],
               "checkpoint_id": _ckpt_id(self.meta), "source": self.name,
               "is_live": False, "meta": self.meta}
        if self.sequential:
            out.update(provenance="recorded_chunk_replay", replayed_from=key)
        else:
            out["provenance"] = self.provenance
        return out


def _ckpt_id(meta: dict[str, Any]) -> str:
    rev = str(meta.get("revision") or "unknown")
    return f"{meta.get('repo_id')}@{rev[:8]}"


class YamAstraReviewSource(AstraReviewSource):
    """The KUKA Astra client, unchanged except for the schema's name."""

    def build_body(self, packet: dict[str, Any],
                   images: list[str] | None = None) -> dict[str, Any]:
        body = super().build_body(packet, images)
        body["text"]["format"]["name"] = "yam_action_review"
        return body


@dataclass
class ShadowGateway:
    """Default gateway. Records what it WOULD command and commands nothing.

    There is no robot handle in this class at all -- shadow is the absence of
    a transport, not a disabled one, exactly as on the KUKA branch.
    """
    name: str = "yam_shadow"
    can_move_robot: bool = False
    emitted: list[dict[str, Any]] = field(default_factory=list)

    def send(self, envelope: CommandEnvelope) -> dict[str, Any]:
        record = {"would_send": envelope.to_log(), "sent": False,
                  "transport": "i2rt CAN joint position command (not opened)",
                  "note": ("SHADOW: no robot handle exists in this gateway. The "
                           "envelope was signed and validated, then logged.")}
        self.emitted.append(record)
        return {"ok": True, "shadow": True, "sent": False, "record": record}
