"""Backend abstraction for the local VLM monitor. No weights are downloaded here.

WHY AN ABSTRACTION RATHER THAN A SERVING ENGINE
The controller must not care whether the monitor is vLLM, llama.cpp, Ollama,
transformers-serve or a mock. Two very different machines have to run the same
typed contract:

ONE MODEL, ON THE JETSON: Qwen3-VL-2B-Instruct.

A 2B VLM is chosen over anything larger because this job is *monitoring* -- read
a scene, report a status -- not reasoning about corrections. The reasoning job
belongs to Astra, and only on escalation. THE JETSON MODEL AND MEMORY ARE
UNKNOWN (see JETSON_COMPUTE); whether a 2B fits, and at what precision, is
~2.2 GB, so memory is not the constraint; LATENCY IS, and it must be measured on
the device rather than assumed.

The monitor cannot live on the A800. That box exposes no HTTP port, has no
Tailscale, cannot reach the public internet, and sits behind a link measured at
~26 KB/s with 20% packet loss -- a single four-frame payload takes ~27 s against
a 3 s budget. A WAN hop inside a motion-gating loop would also mean the arm stops
whenever the link hiccups.

The `MonitorBackend` protocol is unchanged, so a different model remains a
configuration change.

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

#: One model everywhere. The monitor runs ON THE JETSON -- it must, because the
#: A800 is a rented cloud box reachable only over a WAN measured at ~26 KB/s with
#: 20% loss, where a single four-frame payload takes ~27 s against a 3 s budget.
#: A remote monitor was never viable for a loop that gates motion.
#: The board has never been identified. Module/memory decide whether a 2B
#: runs at all, at what precision, and at what latency -- so the 6 s timeout
#: and the pacing derived from it are PLACEHOLDERS, not specifications.
#: Fill these in from the device:
#:     cat /etc/nv_tegra_release ; cat /proc/device-tree/model ; free -g
JETSON_COMPUTE = {
    "board_model": None,          # e.g. "NVIDIA Jetson AGX Orin 64GB"
    "total_memory_gb": None,
    "jetpack_l4t": None,
    "power_mode": None,           # nvpmodel -q  (do NOT change it)
    "measured_2b_latency_s": None,
    "source": "not reported from the device; no value here may be assumed",
}


def compute_is_known() -> tuple[bool, str]:
    """Fail closed: refuse to present latency claims as specifications."""
    missing = [k for k, v in JETSON_COMPUTE.items()
               if v is None and k != "source"]
    if missing:
        return False, ("Jetson compute unidentified (" + ", ".join(missing) +
                       "); the 6 s timeout is a placeholder, not a measurement")
    return True, "Jetson compute reported from the device"


MONITOR_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL = MONITOR_MODEL
EDGE_MODEL = MONITOR_MODEL          # same model everywhere; see JETSON_COMPUTE

#: Dropped as the edge default at the operator's direction. Recorded because the
#: reason matters: the shadow run that showed SmolVLM2-500M returning uniform
#: uncertain/uncertain/confidence-0.0 was OUR wiring -- viewpoints labelled as
#: time, and a trajectory sent as a word count. That run is NOT evidence the
#: model was inadequate, and it was never re-run after the fix. Choosing the 2B
#: is a capacity decision taken without a comparison, not a measured verdict.
RETIRED_EDGE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
RETIRED_REASON = ("operator direction; the 500M was never evaluated on corrected "
                  "inputs, so no quality comparison exists")

#: Documentation only. This module never runs these.
SERVER_EXAMPLES = {
    "jetson_vllm": (
        "vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8020 "
        "--max-model-len 4096 --limit-mm-per-prompt image=4 "
        "--gpu-memory-utilization 0.45"),
    "jetson_llama_cpp": (
        "llama-server -m Qwen3-VL-2B-Instruct-Q8_0.gguf "
        "--mmproj mmproj-Qwen3-VL-2B.gguf --port 8020 -ngl 99"),
    "jetson_transformers": (
        "python -m transformers.serve --model Qwen/Qwen3-VL-2B-Instruct "
        "--port 8020"),
    "measure_first": (
        "Before wiring it in, time one request at the power mode you will "
        "actually run: the 500M took 3.05 s at MODE_30W/612 MHz, and a 2B is "
        "not 4x that but it is not free either. Set monitor_timeout_s from the "
        "measurement, not from hope."),
    "note": ("Start one of these yourself. This package neither installs a "
             "runtime nor fetches weights; it detects whether a service is "
             "reachable and refuses to gate motion when it is not."),
}

MONITOR_SYSTEM_PROMPT = """You are a safety MONITOR for a robot arm. You are not \
a controller and you do not command motion.

You see observation frames, deterministic robot state, and the trajectory the \
policy intends to execute next. Report what you OBSERVE.

Each frame is labelled with its TIME and its CAMERA. Frames from the same \
instant are different viewpoints, not a sequence: judge change over time only \
between frames whose time labels differ. If you are given only one instant, \
report progress from the state and the proposal, and say uncertain about change.

If you cannot see the target, say so rather than guessing.

Return only the JSON object described by the schema. `execute_steps` is a \
REQUEST, not a command: it is clamped independently and a larger number does not \
widen any limit. `confidence` is advisory and never relaxes a bound. `evidence` \
must cite what you actually saw, in at most 120 characters -- one short clause, \
no preamble. Every decision lives in the typed fields, so longer prose adds \
latency without adding information.

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
    """Non-secret settings. A key, if any, is read from the environment by name.

    `timeout_s` IS A DECLARATION ABOUT THE DEVICE, NOT A PREFERENCE. Set it from
    a measurement at the power mode you will actually run. Too low and every
    reading fail-safes to HOLD; too high and the monitor silently stops keeping
    up with the arm while appearing to work.
    """
    endpoint: str = "http://127.0.0.1:8020/v1/chat/completions"
    model: str = MONITOR_MODEL
    api_key_env: str = "LOCAL_VLM_API_KEY"
    timeout_s: float = 6.0
    max_frames: int = 4               # t-1 and t, both cameras
    temperature: float = 0.0
    health_path: str = "/v1/models"

    @classmethod
    def jetson(cls, *, timeout_s: float = 6.0) -> "VlmConfig":
        """The Jetson running Qwen3-VL-2B. Board model UNKNOWN.

        6 s is a STARTING POINT chosen to be honest rather than flattering: the
        500M measured 3.05 s at MODE_30W, a 2B is larger, and a timeout that
        fails every cycle teaches nothing. Measure on the device and tighten it.
        The evidence cap (120 chars) already removed most of the decode cost,
        which is where extra parameters hurt most.
        """
        return cls(model=MONITOR_MODEL, timeout_s=timeout_s, max_frames=4)

    # Retained so existing callers keep working; both now mean the same thing.
    edge = jetson

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

    def build_body(self, *, frames: Sequence[Any], state_text: str,
                   intent_text: str) -> dict[str, Any]:
        """Frames may be bare URLs or (label, url) pairs.

        A LABEL IS NOT COSMETIC. The first version numbered frames by position
        and called index 0 "oldest" -- so when the caller passed base and wrist
        from the SAME tick, the model was told two viewpoints were two moments
        and asked what changed between them. There is no honest answer to that,
        and the model correctly returned uncertain with zero confidence.
        Labels now carry time AND viewpoint, and a caller that supplies only
        viewpoints gets no temporal claim made on its behalf.
        """
        content: list[dict[str, Any]] = []
        items = list(frames)[: self.config.max_frames]
        labelled = [it if isinstance(it, (tuple, list)) and len(it) == 2
                    else (None, it) for it in items]
        has_time = any(lbl for lbl, _ in labelled)
        for i, (label, url) in enumerate(labelled):
            if label:
                tag = f"[{label}]"
            elif has_time:
                tag = f"[frame {i}: unlabelled]"
            else:
                # No temporal information was supplied. Say so rather than
                # inventing an ordering the caller did not claim.
                tag = f"[view {i} of {len(labelled)}, same instant]"
            content.append({"type": "text", "text": tag})
            content.append({"type": "image_url", "image_url": {"url": url}})
        if not has_time and len(labelled) > 1:
            content.append({"type": "text", "text": (
                "NOTE: these are different CAMERA VIEWPOINTS at one instant, not "
                "a time sequence. Do not report change over time from them.")})
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


#: Backend names describe WHERE THE SERVICE IS, not which machine owns the GPU.
#: The old "a800" label was misleading once the monitor moved onto the Jetson:
#: it named a host that cannot serve this at all.
BACKEND_KINDS = ("local", "mock", "unconfigured")
DEPRECATED_KINDS = {"a800": "local", "jetson": "local", "edge": "local",
                    "default": "local", "openai_compatible": "local"}


def make_backend(kind: str = "unconfigured", *,
                 config: VlmConfig | None = None,
                 readings: Sequence[Any] | None = None) -> MonitorBackend:
    kind = DEPRECATED_KINDS.get(kind, kind)
    if kind == "mock":
        return MockMonitorBackend(readings or [])
    if kind == "local":
        return OpenAICompatibleBackend(config or VlmConfig.jetson())
    return UnconfiguredBackend()
