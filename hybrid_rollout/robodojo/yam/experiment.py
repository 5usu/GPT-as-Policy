"""YAM experiment config. Manifests and success assessment are the KUKA ones.

`kuka.experiment` already does the rig-independent work -- episode manifest
validation including video/state alignment, tick-to-frame mapping, and the
rule that success is a predicate or a supervisor, never a model's prose. Those
are re-exported unchanged. Only the config location and the mapping from TOML
sections to the flat safety key space are YAM-specific.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..kuka.experiment import MANIFEST_SCHEMA as MANIFEST_SCHEMA_ID
from ..kuka.experiment import (CameraAssessment, EpisodeManifest, ManifestError,
                               assess_success, frame_for_tick, load_manifest,
                               retry_limits, stop_conditions, validate_manifest)

EXPERIMENT_ROOT = Path(__file__).parent / "experiments"
DEFAULT_EXPERIMENT = "bimanual_blocks"

__all__ = ["CONFIG_KEY_MAP", "CameraAssessment", "DEFAULT_EXPERIMENT",
           "EpisodeManifest", "MANIFEST_SCHEMA_ID", "ManifestError", "assess_success", "flatten_config",
           "frame_for_tick", "load_config", "load_manifest", "local_path", "retry_limits",
           "save_local",
           "stop_conditions", "validate_manifest"]

#: config section.key -> the flat name yam.safety.REQUIRED_CONFIG uses.
CONFIG_KEY_MAP: dict[str, str] = {
    "rig.camera_mapping": "camera_mapping",
    "rig.gripper_limits_left": "gripper_limits_left",
    "rig.gripper_limits_right": "gripper_limits_right",
    "rig.rest_pose": "rest_pose",
    "rig.start_pose": "start_pose",
    "rig.table_workspace": "table_workspace",
    "rig.estop_tested": "estop_tested",
    "limits.max_speed": "max_speed",
    "limits.max_acceleration": "max_acceleration",
    "limits.max_step_displacement": "max_step_displacement",
    "supervision.observation_freshness_s": "observation_freshness_s",
    "supervision.heartbeat_timeout_s": "heartbeat_timeout_s",
    "supervision.success_predicate": "success_predicate",
    "stop_conditions.commanded_observed_tolerance_rad": "commanded_observed_tolerance_rad",
    "astra_direct.authorised_by": "astra_direct_authorised_by",
    "astra_direct.bounded_action_space": "astra_direct_bounded_action_space",
    "astra_direct.dry_run_passed": "astra_direct_dry_run_passed",
    "astra_direct.collision_model": "collision_model",
    # identity: operator facts, not measurements
    "rig.robot_model": "robot_model",
    "rig.rig_id": "rig_id",
    "rig.left_can": "left_can",
    "rig.right_can": "right_can",
}


#: Per-rig measured values live here, next to the committed config, and are
#: git-ignored: the committed file keeps every measured value blank, so a clone
#: onto another rig can never inherit this rig's numbers.
LOCAL_NAME = "rig.local.toml"


def local_path(name: str = DEFAULT_EXPERIMENT, root: Path | None = None) -> Path:
    import os
    env = os.environ.get("YAM_RIG_LOCAL")
    return Path(env) if env else (root or EXPERIMENT_ROOT) / name / LOCAL_NAME


def load_config(name: str = DEFAULT_EXPERIMENT, root: Path | None = None, *,
                local: bool = True) -> dict[str, Any]:
    """The committed config, overlaid key by key with rig.local.toml if present."""
    path = (root or EXPERIMENT_ROOT) / name / "config.toml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment config at {path}")
    cfg = tomllib.loads(path.read_text())
    lp = local_path(name, root)
    if local and lp.exists():
        for section, values in tomllib.loads(lp.read_text()).items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    return cfg


def _toml_value(v: Any) -> str:
    import json
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def save_local(updates: dict[str, dict[str, Any]], name: str = DEFAULT_EXPERIMENT,
               root: Path | None = None, *, note: str = "") -> Path:
    """Merge `updates` ({section: {key: value}}) into rig.local.toml."""
    lp = local_path(name, root)
    cur = tomllib.loads(lp.read_text()) if lp.exists() else {}
    for section, values in updates.items():
        cur.setdefault(section, {}).update(values)
    lines = ["# Measured on THIS rig. Git-ignored; never commit it.",
             f"# last written by yam.cli {note}".rstrip(), ""]
    for section, values in cur.items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        lines.append("")
    lp.write_text("\n".join(lines))
    return lp


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map to the flat safety key space. Empty string stays empty -- unset."""
    flat: dict[str, Any] = {}
    for dotted, flat_key in CONFIG_KEY_MAP.items():
        section, key = dotted.split(".", 1)
        flat[flat_key] = (cfg.get(section) or {}).get(key)
    return flat
