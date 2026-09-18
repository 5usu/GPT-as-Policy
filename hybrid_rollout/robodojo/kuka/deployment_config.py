"""Versioned deployment configuration: every measured value, null by default.

NOTHING HERE IS EVER INFERRED OR FABRICATED.
Each field below is a quantity a human must measure on the real cell. The
default is null, `validate` names every one that is still null, and there is no
code path that fills one in from a similar robot, a previous cell, or a
plausible range. A wrong number that looks measured is more dangerous than an
obviously missing one.

`report()` produces a machine-readable GO/NO-GO naming every failed gate, so the
answer to "can we run yet" is a file rather than a judgement call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kuka.deployment_config.v1"

#: field -> (section, what to measure, why it matters)
FIELDS: dict[str, tuple[str, str, str]] = {
    # --- geometry -----------------------------------------------------------
    "handle_pose": ("geometry", "dishwasher door handle pose in robot base frame",
                    "the target of the whole task"),
    "door_hinge_axis": ("geometry", "hinge origin + direction, base frame",
                        "defines the arc the door may travel"),
    "door_open_region": ("geometry", "expected door sweep volume",
                         "motion outside it is a stop condition"),
    "table_workspace": ("geometry", "allowed Cartesian volume",
                        "bounds the whole task"),
    # --- calibration --------------------------------------------------------
    "camera_intrinsics": ("calibration", "K + distortion for base and wrist",
                          "without it pixels cannot become geometry"),
    "camera_extrinsics": ("calibration", "camera poses in robot base frame",
                          "without it a camera observation cannot be located"),
    "tcp_transform": ("calibration", "$TOOL read off the controller",
                      "flange != tool tip; FK is flange-frame until this exists"),
    "base_frame": ("calibration", "verified base frame definition",
                   "everything spatial is expressed relative to it"),
    "gripper_polarity": ("calibration", "which commanded value is OPEN, which CLOSED",
                         "reversed means gripping when you meant to release"),
    # --- limits -------------------------------------------------------------
    "force_torque_limits": ("limits", "load at which motion must abort",
                            "the only contact-aware stop for a contact-rich task"),
    "max_speed": ("limits", "commanded speed ceiling for this task",
                  "task-specific, lower than the machine maximum"),
    "max_acceleration": ("limits", "commanded acceleration ceiling", "as above"),
    "max_step_displacement": ("limits", "largest displacement per supervised step",
                              "bounds how wrong one step can be"),
    # --- OTG limits (feed Ruckig) -------------------------------------------
    "max_velocity_deg_s": ("otg", "per-joint velocity limit, 6 values",
                           "Ruckig refuses to generate without it"),
    "max_acceleration_deg_s2": ("otg", "per-joint acceleration limit, 6 values",
                                "Ruckig refuses to generate without it"),
    "max_jerk_deg_s3": ("otg", "per-joint jerk limit, 6 values",
                        "the difference between smooth and violent motion"),
    # --- supervision --------------------------------------------------------
    "observation_freshness_s": ("supervision", "maximum observation age",
                                "a stale observation describes a pose already left"),
    "heartbeat_timeout_s": ("supervision", "supervisor heartbeat timeout",
                            "detects a dead supervisor"),
    "success_predicate": ("supervision", "a MEASURABLE predicate for success",
                          "model prose is never accepted as success"),
    "commanded_observed_tolerance_deg": ("supervision",
                                         "per-joint commanded-vs-measured tolerance",
                                         "unset makes the check unevaluable, which stops"),
    "tolerance_severe_multiplier": ("supervision",
                                    "multiple of tolerance that latches FAULT immediately",
                                    "separates lag from collision"),
    "tolerance_persistence_cycles": ("supervision",
                                     "consecutive breaches before latching FAULT",
                                     "separates lag from collision"),
    # --- devices ------------------------------------------------------------
    "camera_device_mapping": ("devices", 'e.g. {"base": 0, "wrist": 2}',
                              "node order is not stable; guessing pairs the wrong view"),
    "gripper_device_id": ("devices", "verified identity of /dev/ttyUSB0",
                          "USB enumeration order is not stable"),
    "gripper_open_close_limits": ("devices", "travel limits in device units",
                                  "bounds how far it can close on a hand"),
    "gripper_speed_force_limits": ("devices", "speed and force ceilings",
                                   "a contact-rich task needs a force ceiling"),
    "gripper_safe_action": ("devices", "what the gripper does on arm HOLD/FAULT/E-stop",
                            "an arm stop does NOT stop a Modbus gripper"),
    # --- astra-direct, JOINT space (this cell's controller takes AK.A1..A6) --
    "direct_max_step_deg": ("astra_direct_joint",
                            "per-joint max change between consecutive proposed points",
                            "the only per-step bound once pi0.5 is out of the loop"),
    "direct_max_total_excursion_deg": ("astra_direct_joint",
                                       "per-joint max distance from the MEASURED pose",
                                       "stops a run of small steps walking somewhere far"),
    "direct_max_steps": ("astra_direct_joint", "max proposed points per decision",
                         "bounds how much is committed on one judgement"),
    "direct_authorised_by": ("astra_direct_joint", "named human who authorised it",
                             "accountability for removing the policy's sanity floor"),
    "direct_shadow_campaign": ("astra_direct_joint",
                               "reference to a completed shadow campaign on this task",
                               "evidence before authority"),
    # --- astra-direct, CARTESIAN (upstream's form; blocked by the interface) -
    "astra_direct_authorised_by": ("astra_direct", "named human", "accountability"),
    "astra_direct_bounded_action_space": ("astra_direct", "explicit bounds",
                                          "an unbounded self-proposal is the thing to prevent"),
    "astra_direct_dry_run_passed": ("astra_direct", "reference to a shadow campaign",
                                    "evidence before authority"),
    "collision_model": ("astra_direct", "cell + self collision model",
                        "no local check exists without it"),
}

GATE_SETS = {
    "recorded_replay": (),
    "shadow_review": ("camera_device_mapping",),
    "hold_handshake": (),
    "supervised_student_prefix": tuple(
        k for k, (sec, _, _) in FIELDS.items()
        if sec not in ("astra_direct", "astra_direct_joint")),
    # Joint-space direct: everything the supervised prefix needs, PLUS its own
    # bounds. Reachable through this controller.
    "astra_direct_joint": tuple(
        k for k, (sec, _, _) in FIELDS.items() if sec != "astra_direct"),
    # Cartesian direct: upstream's form. Blocked by the RSI receive config.
    "astra_direct": tuple(FIELDS),
}


def blank() -> dict[str, Any]:
    """A configuration with every measured value null. The honest starting point."""
    return {"schema": SCHEMA_VERSION, "cell": None, "measured_by": None,
            "measured_at": None,
            "values": {k: None for k in FIELDS},
            "_reference_only": {
                "note": ("the deployed teleop tuning, for the engineer to CONFIRM "
                         "or replace. Never read automatically."),
                "max_velocity_deg_s": [85.0, 40.0, 125.0, 125.0, 125.0, 320.0],
                "max_acceleration_deg_s2": [800.0, 250.0, 500.0, 1200.0, 1200.0, 2500.0],
                "max_jerk_deg_s3": [12000.0, 12000.0, 15000.0, 20000.0, 20000.0, 40000.0],
                "source": "KUKA/teleoperation/udp_teleoperate.py:135-141"}}


def load(path: str | Path) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA_VERSION}, got {d.get('schema')!r}")
    return d


def values(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("values") or {})


def missing(cfg: dict[str, Any], gate: str) -> list[str]:
    v = values(cfg)
    required = GATE_SETS.get(gate, ())
    return sorted(k for k in required if v.get(k) in (None, "", [], {}))


def report(cfg: dict[str, Any], *,
           extra_gates: dict[str, tuple[bool, list[str]]] | None = None
           ) -> dict[str, Any]:
    """Machine-readable GO/NO-GO. Every failed gate is named."""
    out: dict[str, Any] = {"schema": "kuka.preflight_report.v1",
                           "config_schema": cfg.get("schema"),
                           "cell": cfg.get("cell"),
                           "measured_by": cfg.get("measured_by"),
                           "gates": {}}
    for gate in GATE_SETS:
        miss = missing(cfg, gate)
        blockers = [f"missing measured value: {m}" for m in miss]
        if extra_gates and gate in extra_gates:
            ok_x, bx = extra_gates[gate]
            if not ok_x:
                blockers.extend(bx)
        out["gates"][gate] = {
            "decision": "GO" if not blockers else "NO-GO",
            "blockers": blockers,
            "n_missing_values": len(miss),
        }
    out["overall"] = ("GO" if all(g["decision"] == "GO"
                                  for g in out["gates"].values()) else "NO-GO")
    out["summary"] = {g: v["decision"] for g, v in out["gates"].items()}
    return out


def checklist(cfg: dict[str, Any] | None = None) -> str:
    """Printable checklist for the deployment engineer."""
    v = values(cfg) if cfg else {k: None for k in FIELDS}
    lines = ["KUKA DISHWASHER CELL -- MEASURED VALUE CHECKLIST",
             f"schema {SCHEMA_VERSION}", ""]
    section = None
    for key, (sec, what, why) in FIELDS.items():
        if sec != section:
            section = sec
            lines.append(f"[{sec.upper()}]")
        mark = "x" if v.get(key) not in (None, "", [], {}) else " "
        lines.append(f"  [{mark}] {key}")
        lines.append(f"        measure: {what}")
        lines.append(f"        why    : {why}")
    lines += ["", "Every unchecked box blocks at least one gate.",
              "Nothing in this system will fill one in for you."]
    return "\n".join(lines)
