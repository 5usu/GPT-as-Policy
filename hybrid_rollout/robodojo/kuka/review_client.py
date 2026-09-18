"""Astra review client: streaming, heartbeated, bounded retries, HOLD on timeout.

WHY STREAMING
A non-streaming call is indistinguishable from a hung one until it returns or
times out. Measured on the real endpoint, ~770 KB bodies produced 180 s hangs on
roughly every other call in a back-to-back batch. Streaming lets the caller see
tokens arriving and emit a heartbeat, so a supervisor can tell "still thinking"
from "dead" -- which matters when a robot is holding position waiting.

TIMEOUT IS A HOLD, NOT AN ERROR TO PAPER OVER
If the reviewer does not answer in time the result is `hold`: keep the arm where
it is, command nothing. That is a decision, and it is recorded as one. The one
thing it must never be is an empty result that a caller might read as "no
objection".

IDEMPOTENT REQUEST IDS
The id is derived from the request body, so a retry of the SAME question carries
the SAME id and a different question cannot accidentally reuse one. Retries are
bounded and only cover transport-level failures; a reviewer that answered is
never re-asked, because re-asking until you like the answer is how a batch stops
meaning anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.review_client.v1"

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_HEARTBEAT_S = 5.0
RETRYABLE = ("timeout", "transport_error", "http_429", "http_500", "http_502",
             "http_503", "http_504")


class ReviewOutcome(str, Enum):
    DECIDED = "decided"
    HOLD_TIMEOUT = "hold_timeout"
    HOLD_ERROR = "hold_error"
    HOLD_UNCONFIGURED = "hold_unconfigured"
    DRY_RUN = "dry_run"


@dataclass
class ReviewResult:
    outcome: ReviewOutcome
    request_id: str
    body_sha256_12: str
    decision: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    latency_s: float = 0.0
    usage: dict[str, Any] | None = None
    heartbeats: int = 0
    bytes_streamed: int = 0

    @property
    def is_hold(self) -> bool:
        return self.outcome is not ReviewOutcome.DECIDED

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["schema"] = SCHEMA
        d["is_hold"] = self.is_hold
        if self.is_hold:
            d["hold_meaning"] = ("no usable decision. The arm holds position and "
                                 "nothing is commanded. This is NOT 'no objection'.")
        return d


def idempotent_request_id(body: dict[str, Any], *, prefix: str = "kuka") -> str:
    """Same question -> same id; different question -> different id."""
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    return f"{prefix}-{digest[:24]}"


def classify_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "timed out" in low or "timeout" in low:
        return "timeout"
    for code in ("429", "500", "502", "503", "504"):
        if f"http {code}" in low:
            return f"http_{code}"
    if "401" in low or "403" in low:
        return "http_auth"
    return "transport_error"


class StreamingReviewClient:
    """Off by default. `enabled=False` or a dry run never opens a connection."""

    name = "astra_streaming"

    def __init__(self, *, base_url: str, model: str, api_key_env: str,
                 enabled: bool = False, dry_run: bool = True,
                 reasoning: str | None = None, store: bool = False,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 heartbeat_s: float = DEFAULT_HEARTBEAT_S,
                 transport: Callable | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.enabled = bool(enabled)
        self.dry_run = bool(dry_run)
        self.reasoning = reasoning
        self.store = store
        self.timeout_s = timeout_s
        self.max_attempts = max(1, int(max_attempts))
        self.heartbeat_s = heartbeat_s
        self.transport = transport

    @property
    def is_live(self) -> bool:
        return self.enabled and not self.dry_run

    def preflight(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "client constructed with enabled=False"
        if not os.environ.get(self.api_key_env):
            return False, f"${self.api_key_env} is empty"
        return True, "ready"

    def build_body(self, packet: dict[str, Any],
                   images: Sequence[str] | None = None) -> dict[str, Any]:
        content: list[Any] = [{"type": "input_text", "text": packet["user_text"]}]
        for url in images or packet.get("image_data_urls") or []:
            content.append({"type": "input_image", "image_url": url})
        body: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "system", "content": packet["system"]},
                      {"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "kuka_action_review",
                                "schema": packet["response_schema"], "strict": True}},
            "store": bool(self.store), "stream": True}
        if self.reasoning:
            body["reasoning"] = {"effort": self.reasoning}
        return body

    def review(self, packet: dict[str, Any], *,
               on_heartbeat: Callable[[dict[str, Any]], None] | None = None
               ) -> ReviewResult:
        body = self.build_body(packet)
        sha = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        rid = idempotent_request_id(body)

        if self.dry_run or not self.enabled:
            return ReviewResult(
                ReviewOutcome.DRY_RUN if self.dry_run else ReviewOutcome.HOLD_UNCONFIGURED,
                rid, sha, error=None if self.dry_run else self.preflight()[1],
                bytes_streamed=len(json.dumps(body)))
        ok, why = self.preflight()
        if not ok:
            return ReviewResult(ReviewOutcome.HOLD_UNCONFIGURED, rid, sha, error=why)

        last_err, attempts = None, 0
        t_start = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            hb = {"n": 0}

            def beat(info: dict[str, Any]) -> None:
                hb["n"] += 1
                if on_heartbeat:
                    on_heartbeat({"request_id": rid, "attempt": attempt,
                                  "elapsed_s": round(time.monotonic() - t_start, 1),
                                  **info})
            try:
                text, usage, nbytes = self._stream(body, rid, beat)
            except Exception as exc:                            # noqa: BLE001
                kind = classify_error(exc)
                last_err = f"{kind}: {exc}"[:300]
                if kind not in RETRYABLE or attempt == self.max_attempts:
                    break
                time.sleep(min(2.0 * attempt, 10.0))
                continue
            latency = time.monotonic() - t_start
            if text is None:
                last_err = "no text in stream"
                break
            try:
                decision = json.loads(text)
            except Exception:
                last_err = "response was not valid JSON"
                break
            return ReviewResult(ReviewOutcome.DECIDED, rid, sha, decision=decision,
                                attempts=attempts, latency_s=round(latency, 2),
                                usage=usage, heartbeats=hb["n"],
                                bytes_streamed=nbytes)
        outcome = (ReviewOutcome.HOLD_TIMEOUT
                   if last_err and last_err.startswith("timeout")
                   else ReviewOutcome.HOLD_ERROR)
        return ReviewResult(outcome, rid, sha, error=last_err, attempts=attempts,
                            latency_s=round(time.monotonic() - t_start, 2))

    def _stream(self, body: dict[str, Any], rid: str,
                beat: Callable[[dict[str, Any]], None]):
        """SSE read loop. Emits a heartbeat at most every heartbeat_s."""
        if self.transport is not None:
            return self.transport(body, rid, beat)
        import urllib.request
        req = urllib.request.Request(
            self.base_url, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {os.environ[self.api_key_env]}",
                     "Content-Type": "application/json", "Accept": "text/event-stream",
                     "Idempotency-Key": rid},
            method="POST")
        chunks: list[str] = []
        usage: dict[str, Any] | None = None
        nbytes = 0
        last_beat = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            for raw in r:
                nbytes += len(raw)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue
                if isinstance(ev.get("delta"), str):
                    chunks.append(ev["delta"])
                elif isinstance(ev.get("text"), str):
                    chunks.append(ev["text"])
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                now = time.monotonic()
                if now - last_beat >= self.heartbeat_s:
                    last_beat = now
                    beat({"bytes": nbytes, "chars": sum(len(c) for c in chunks)})
        return ("".join(chunks) or None), usage, nbytes
