"""Experiment config + recorded-episode manifest handling.

Two jobs, both about refusing to proceed on unverified ground:

  1. Flatten an experiment config into the flat key space `safety.missing_config`
     checks, WITHOUT inventing anything. An unset value stays unset.
  2. Validate a recorded episode manifest, including VIDEO/STATE ALIGNMENT. A
     review that reasons over frame k while the state row for frame k belongs to
     a different instant is not a review of anything; misalignment is an error,
     not a warning.

SUCCESS IS NOT A MODEL OPINION.
`assess_success` accepts a camera assessment (evidence frames + confidence) but
will not report success from it. Reportable success requires a configured
measurable predicate or an explicit supervisor confirmation. Prose, however
confident, yields `unconfirmed`.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "hybrid_rollout.robodojo.kuka.experiment.v1"
MANIFEST_SCHEMA = "hybrid_rollout.robodojo.kuka.episode_manifest.v1"

EXPERIMENT_ROOT = Path(__file__).parent / "experiments"

#: config section.key -> the flat name safety.REQUIRED_CONFIG uses.
CONFIG_KEY_MAP: dict[str, str] = {
    "geometry.handle_pose": "handle_pose",
    "geometry.door_hinge_axis": "door_hinge_axis",
    "geometry.door_open_region": "door_open_region",
    "geometry.table_workspace": "table_workspace",
    "calibration.camera_intrinsics": "camera_intrinsics",
    "calibration.camera_extrinsics": "camera_extrinsics",
    "calibration.tcp_transform": "tcp_transform",
    "calibration.base_frame": "base_frame",
    "calibration.gripper_polarity": "gripper_polarity",
    "limits.force_torque_limits": "force_torque_limits",
    "limits.max_speed": "max_speed",
    "limits.max_acceleration": "max_acceleration",
    "limits.max_step_displacement": "max_step_displacement",
    "supervision.observation_freshness_s": "observation_freshness_s",
    "supervision.heartbeat_timeout_s": "heartbeat_timeout_s",
    "supervision.success_predicate": "success_predicate",
    "astra_direct.authorised_by": "astra_direct_authorised_by",
    "astra_direct.bounded_action_space": "astra_direct_bounded_action_space",
    "astra_direct.dry_run_passed": "astra_direct_dry_run_passed",
    "astra_direct.collision_model": "collision_model",
}


class ManifestError(Exception):
    """A recorded episode cannot be trusted as described."""


def load_config(name: str = "dishwasher_door_open",
                root: Path | None = None) -> dict[str, Any]:
    path = (root or EXPERIMENT_ROOT) / name / "config.toml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment config at {path}")
    return tomllib.loads(path.read_text())


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map to the flat safety key space. Empty string stays empty -- unset."""
    flat: dict[str, Any] = {}
    for dotted, flat_key in CONFIG_KEY_MAP.items():
        section, key = dotted.split(".", 1)
        flat[flat_key] = (cfg.get(section) or {}).get(key)
    return flat


def stop_conditions(cfg: dict[str, Any]) -> list[str]:
    return list((cfg.get("stop_conditions") or {}).get("stop_on") or [])


def retry_limits(cfg: dict[str, Any]) -> dict[str, int]:
    sc = cfg.get("stop_conditions") or {}
    return {"max_retries_per_subgoal": int(sc.get("max_retries_per_subgoal", 0)),
            "max_consecutive_no_progress": int(sc.get("max_consecutive_no_progress", 0))}


# ------------------------------------------------------------------- manifests
@dataclass
class EpisodeManifest:
    raw: dict[str, Any]

    @property
    def episode_id(self) -> str:
        return str(self.raw.get("episode_id", ""))

    @property
    def instruction(self) -> str:
        return str((self.raw.get("task") or {}).get("instruction", ""))

    @property
    def videos(self) -> list[dict[str, Any]]:
        return list(self.raw.get("videos") or [])

    @property
    def outcome(self) -> dict[str, Any]:
        return dict(self.raw.get("outcome") or {})

    def to_log(self) -> dict[str, Any]:
        v = self.raw.get("videos") or []
        return {"episode_id": self.episode_id, "instruction": self.instruction,
                "cameras": [c.get("camera") for c in v],
                "checkpoint": (self.raw.get("checkpoint") or {}).get("id"),
                "outcome": self.outcome}


def load_manifest(path: str | Path) -> EpisodeManifest:
    return EpisodeManifest(json.loads(Path(path).read_text()))


def validate_manifest(m: EpisodeManifest, *,
                      alignment_tolerance_s: float = 0.050) -> list[str]:
    """Structural + alignment checks. Returns problems; empty means usable."""
    problems: list[str] = []
    raw = m.raw
    if raw.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema is {raw.get('schema')!r}, expected {MANIFEST_SCHEMA!r}")
    for field_name in ("episode_id", "task", "checkpoint", "timestamps", "videos",
                       "state", "trajectory", "calibration", "outcome"):
        if field_name not in raw:
            problems.append(f"missing required field {field_name!r}")
    if problems:
        return problems

    videos, state = m.videos, raw["state"]
    if not videos:
        problems.append("no videos listed")
    cams = [v.get("camera") for v in videos]
    if len(set(cams)) != len(cams):
        problems.append(f"duplicate camera entries: {cams}")

    ts = raw["timestamps"]
    hz = float(ts.get("control_hz") or 0)
    if hz <= 0:
        problems.append("timestamps.control_hz must be positive")

    # --- alignment: this is the check that actually matters ------------------
    state_epoch = state.get("first_row_epoch")
    n_rows = int(state.get("n_rows") or 0)
    for v in videos:
        v_epoch = v.get("first_frame_epoch")
        if v_epoch is None or state_epoch is None:
            problems.append(f"camera {v.get('camera')}: missing alignment timestamp")
            continue
        skew = abs(float(v_epoch) - float(state_epoch))
        if skew > alignment_tolerance_s:
            problems.append(
                f"camera {v.get('camera')}: first frame is {skew:.3f}s from the "
                f"first state row (tolerance {alignment_tolerance_s}s); frames "
                f"and state rows do not describe the same instants")
        fps = float(v.get("fps") or 0)
        n_frames = int(v.get("n_frames") or 0)
        if fps > 0 and hz > 0 and n_rows > 0 and n_frames > 0:
            v_dur, s_dur = n_frames / fps, n_rows / hz
            if abs(v_dur - s_dur) > max(alignment_tolerance_s, 0.5 / hz):
                problems.append(
                    f"camera {v.get('camera')}: video spans {v_dur:.3f}s but state "
                    f"spans {s_dur:.3f}s; streams have different durations")
    for ck in ("path", "sha256"):
        if not state.get(ck):
            problems.append(f"state.{ck} missing")
        if not (raw.get("trajectory") or {}).get(ck):
            problems.append(f"trajectory.{ck} missing")
    if (raw.get("trajectory") or {}).get("provenance") != "recorded_demo":
        problems.append("trajectory.provenance must be 'recorded_demo'")
    return problems


def frame_for_tick(m: EpisodeManifest, camera: str, tick: int) -> int:
    """Frame index matching a control tick, from the manifest's own rates."""
    hz = float((m.raw.get("timestamps") or {}).get("control_hz") or 0)
    v = next((x for x in m.videos if x.get("camera") == camera), None)
    if v is None:
        raise ManifestError(f"no camera {camera!r} in manifest")
    fps = float(v.get("fps") or 0)
    if hz <= 0 or fps <= 0:
        raise ManifestError("cannot map ticks to frames without both rates")
    idx = int(round(tick * fps / hz))
    n = int(v.get("n_frames") or 0)
    if not 0 <= idx < n:
        raise ManifestError(f"tick {tick} -> frame {idx} outside 0..{n - 1}")
    return idx


# --------------------------------------------------------------------- success
@dataclass
class CameraAssessment:
    """What a vision review returns. Evidence and confidence are REQUIRED so a
    claim can be checked; neither is sufficient to declare success."""
    progress: str
    evidence_frames: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    prose: str = ""

    def valid(self) -> tuple[bool, str]:
        if not self.evidence_frames:
            return False, "no evidence frames cited"
        if self.confidence is None:
            return False, "no confidence reported"
        if not 0.0 <= float(self.confidence) <= 1.0:
            return False, f"confidence {self.confidence} outside [0,1]"
        return True, ""

    def to_log(self) -> dict[str, Any]:
        ok, why = self.valid()
        return {"progress": self.progress, "confidence": self.confidence,
                "evidence_frames": self.evidence_frames, "prose": self.prose[:500],
                "assessment_valid": ok, "assessment_problem": why,
                "sufficient_for_success": False,
                "note": ("a camera assessment is evidence for a human, never a "
                         "success declaration; see assess_success")}


def assess_success(*, camera: CameraAssessment | None,
                   predicate_value: Any = None,
                   predicate_config: Any = None,
                   supervisor_confirmed: bool | None = None) -> dict[str, Any]:
    """Reportable success. Model prose alone can never produce success=True."""
    reasons: list[str] = []
    if supervisor_confirmed is True:
        return {"success": True, "source": "supervisor_confirmed",
                "camera": camera.to_log() if camera else None,
                "reasons": ["explicit supervisor confirmation"]}
    if predicate_config in (None, "", [], {}):
        reasons.append("no measurable success_predicate configured")
    elif predicate_value is None:
        reasons.append("success_predicate configured but not evaluated")
    else:
        return {"success": bool(predicate_value), "source": "measured_predicate",
                "predicate_value": predicate_value,
                "camera": camera.to_log() if camera else None,
                "reasons": ["measured predicate evaluated"]}
    if camera is not None:
        reasons.append(
            "camera assessment present but NOT accepted as a success source: "
            "model prose and confidence are evidence, not measurement")
    return {"success": None, "source": "unknown", "reasons": reasons,
            "camera": camera.to_log() if camera else None}
