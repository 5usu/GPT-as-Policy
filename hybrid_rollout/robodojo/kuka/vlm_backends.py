"""Backend abstraction for the local VLM monitor. No weights are downloaded here.

WHY AN ABSTRACTION RATHER THAN A SERVING ENGINE
The controller must not care whether the monitor is vLLM, llama.cpp, Ollama,
transformers-serve or a mock. Two very different machines have to run the same
typed contract:

  A800 (default)  Qwen3-VL-2B-Instruct. Small enough to sit beside other work on
                  an 80 GB card and still answer at a few Hz; a 2B VLM is chosen
                  over anything larger because this job is *monitoring* -- read a
                  scene, report a status -- not reasoning about corrections. The
                  reasoning job belongs to Astra, and only on escalation.

  Jetson (edge)   SmolVLM2-500M-Video-Instruct. Video-instruct matters: the
                  monitor's decisions depend on CHANGE between frames, and a
                  model built for a single image tends to describe a still.
                  500M keeps headroom on a board already running the RSI loop.

Both speak the same `MonitorBackend` protocol, so swapping one for the other is a
configuration change and the controller is untouched.

NOTHING HERE DOWNLOADS WEIGHTS OR INSTALLS A RUNTIME. The HTTP client assumes an
OpenAI-compatible server the operator started themselves; `probe()` reports
whether one is actually there. See SERVER_EXAMPLES for the commands, which are
documentation, not something this module runs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

from .vlm_monitor import MonitorReading, MonitorRejected, parse_reading, response_schema

SCHEMA = "hybrid_rollout.robodojo.kuka.vlm_backends.v1"

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
EDGE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

#: Documentation only. This module never runs these.
SERVER_EXAMPLES = {
    "a800_vllm": (
        "vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8020 "
        "--max-model-len 8192 --limit-mm-per-prompt image=4"),
    "a800_transformers": (
        "python -m transformers.serve --model Qwen/Qwen3-VL-2B-Instruct "
        "--port 8020"),
    "jetson_llama_cpp": (
        "llama-server -m SmolVLM2-500M-Video-Instruct-Q8_0.gguf "
        "--mmproj mmproj-SmolVLM2-500M.gguf --port 8020"),
    "note": ("Start one of these yourself. This package neither installs a "
             "runtime nor fetches weights; it detects whether a service is "
             "reachable and refuses to gate motion when it is not."),
}

MONITOR_SYSTEM_PROMPT = """You are a safety MONITOR for a robot arm. You are not \
a controller and you do not command motion.

You see recent observation frames (oldest first) and deterministic robot state, \
plus the trajectory the policy intends to execute next. Report what you OBSERVE.

Judge CHANGE between the frames, not a single still. If you cannot see the \
target, say so rather than guessing.

Return only the JSON object described by the schema. `execute_steps` is a \
REQUEST, not a command: it is clamped independently and a larger number does not \
widen any limit. `confidence` is advisory and never relaxes a bound. `evidence` \
must cite what you actually saw; a decision without stated evidence is not \
auditable.

Set escalate=true only for persistent evidence of execution failure or a \
trajectory that pursues the wrong subgoal. Uncertainty alone is not escalation; \
report it as uncertain and let the deterministic layer shorten the prefix."""


@dataclass
class BackendProbe:
    available: bool
    reason: str
    model: str | None = None
    endpoint: str | None = None
    latency_s: float | None = None

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d


class MonitorBackend(Protocol):
    name: str
    model: str
    is_mock: bool

    def probe(self) -> BackendProbe: ...

    def observe(self, *, frames: Sequence[str], state_text: str,
                intent_text: str, timeout_s: float) -> MonitorReading: ...


class UnconfiguredBackend:
    """Default. Refuses loudly instead of pretending a monitor exists."""

    name = "unconfigured"
    model = ""
    is_mock = False

    def probe(self) -> BackendProbe:
        return BackendProbe(False, "no monitor backend configured")

    def observe(self, **_kw) -> MonitorReading:
        raise MonitorRejected("no_backend", "no monitor backend configured")


class MockMonitorBackend:
    """Scripted readings for offline tests. Never mistaken for a real monitor:
    `is_mock` is True and every reading carries backend='mock'."""

    name = "mock"
    is_mock = True

    def __init__(self, readings: Sequence[Any], *, model: str = "mock",
                 latency_s: float = 0.01, fail_with: str | None = None) -> None:
        self._readings = list(readings)
        self._i = 0
        self.model = model
        self.latency_s = latency_s
        self.fail_with = fail_with

    def probe(self) -> BackendProbe:
        return BackendProbe(self.fail_with is None,
                            self.fail_with or "mock backend ready",
                            self.model, "mock://", 0.0)

    def observe(self, **_kw) -> MonitorReading:
        if self.fail_with:
            raise MonitorRejected("backend_error", self.fail_with)
        if self._i >= len(self._readings):
            raise MonitorRejected("mock_exhausted", "no scripted reading left")
        payload = self._readings[self._i]
        self._i += 1
        return parse_reading(payload, backend="mock", latency_s=self.latency_s)


@dataclass
class VlmConfig:
    """Non-secret settings. A key, if any, is read from the environment by name."""
    endpoint: str = "http://127.0.0.1:8020/v1/chat/completions"
    model: str = DEFAULT_MODEL
    api_key_env: str = "LOCAL_VLM_API_KEY"
    timeout_s: float = 2.0
    max_frames: int = 3
    temperature: float = 0.0
    health_path: str = "/v1/models"

    @classmethod
    def edge(cls) -> "VlmConfig":
        return cls(model=EDGE_MODEL, timeout_s=3.0, max_frames=2)

    def to_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.update({"schema": SCHEMA, "key_present": bool(
            os.environ.get(self.api_key_env))})
        return d


class OpenAICompatibleBackend:
    """Talks to an OpenAI-compatible chat-completions endpoint the operator ran.

    Deliberately thin: this is the widest-supported local-serving interface
    (vLLM, llama.cpp server, Ollama, transformers-serve all expose it), so the
    controller is not coupled to one engine.
    """

    name = "openai_compatible"
    is_mock = False

    def __init__(self, config: VlmConfig | None = None, *, transport=None) -> None:
        self.config = config or VlmConfig()
        self.model = self.config.model
        self.transport = transport

    def _base(self) -> str:
        e = self.config.endpoint
        return e.split("/v1/")[0] if "/v1/" in e else e.rsplit("/", 1)[0]

    def probe(self) -> BackendProbe:
        """Is a service actually there? Called BEFORE motion is gated."""
        url = self._base() + self.config.health_path
        t0 = time.monotonic()
        try:
            if self.transport is not None:
                self.transport(url, None, 1.0)
            else:
                import urllib.request
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    r.read(1)
        except Exception as exc:
            return BackendProbe(
                False,
                f"no monitor service at {url}: {type(exc).__name__}. Start one "
                f"(see SERVER_EXAMPLES); this package will not gate motion on a "
                f"monitor that is not there.",
                self.model, self.config.endpoint)
        return BackendProbe(True, "monitor service reachable", self.model,
                            self.config.endpoint,
                            round(time.monotonic() - t0, 4))

    def build_body(self, *, frames: Sequence[str], state_text: str,
                   intent_text: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for i, url in enumerate(list(frames)[: self.config.max_frames]):
            label = "oldest" if i == 0 else (
                "current" if i == len(frames) - 1 else f"t-{len(frames)-1-i}")
            content.append({"type": "text", "text": f"[frame {i}: {label}]"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text",
                        "text": f"ROBOT STATE (deterministic):\n{state_text}"})
        content.append({"type": "text",
                        "text": f"INTENDED NEXT TRAJECTORY:\n{intent_text}"})
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": MONITOR_SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "kuka_monitor_status", "strict": True,
                "schema": response_schema()}},
        }

    def observe(self, *, frames: Sequence[str], state_text: str,
                intent_text: str, timeout_s: float | None = None) -> MonitorReading:
        body = self.build_body(frames=frames, state_text=state_text,
                               intent_text=intent_text)
        timeout = timeout_s or self.config.timeout_s
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        t0 = time.monotonic()
        try:
            if self.transport is not None:
                raw = self.transport(self.config.endpoint, body, timeout)
            else:
                import urllib.request
                req = urllib.request.Request(
                    self.config.endpoint, data=json.dumps(body).encode(),
                    headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = json.loads(r.read().decode())
        except Exception as exc:
            raise MonitorRejected(
                "backend_error", f"{type(exc).__name__}: {exc}"[:200]) from None
        latency = time.monotonic() - t0
        try:
            text = raw["choices"][0]["message"]["content"]
        except Exception:
            raise MonitorRejected("no_content",
                                  "response had no message content") from None
        return parse_reading(text, backend=f"{self.name}:{self.model}",
                             latency_s=round(latency, 4))


def make_backend(kind: str = "unconfigured", *,
                 config: VlmConfig | None = None,
                 readings: Sequence[Any] | None = None) -> MonitorBackend:
    if kind == "mock":
        return MockMonitorBackend(readings or [])
    if kind in ("a800", "default", "openai_compatible"):
        return OpenAICompatibleBackend(config or VlmConfig())
    if kind in ("jetson", "edge"):
        return OpenAICompatibleBackend(config or VlmConfig.edge())
    return UnconfiguredBackend()
