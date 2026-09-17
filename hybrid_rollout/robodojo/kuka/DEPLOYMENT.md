# Deployment handoff — KUKA Astra review gateway

For the deployment engineer. Everything here is offline-verified; **no robot,
simulator, GPU model or paid API call has been made.**

## What you must supply before anything can be armed

`hybrid_rollout/robodojo/kuka/experiments/dishwasher_door_open/config.toml` ships
with **22 empty values**. Each is a physical quantity nobody has measured. They
are empty on purpose: a plausible default would look like knowledge. Arming
refuses and names them until each is filled.

### Cell geometry (robot base frame)
| key | what to measure |
|---|---|
| `geometry.handle_pose` | dishwasher door handle pose |
| `geometry.door_hinge_axis` | hinge origin + direction |
| `geometry.door_open_region` | expected door sweep; motion outside it stops the run |
| `geometry.table_workspace` | allowed Cartesian volume |

### Calibration
| key | what to measure |
|---|---|
| `calibration.camera_intrinsics` | K + distortion, base and wrist |
| `calibration.camera_extrinsics` | camera poses in base frame |
| `calibration.tcp_transform` | **verified `$TOOL`/TCP** (flange → tool tip) |
| `calibration.base_frame` | verified base frame definition |
| `calibration.gripper_polarity` | which commanded value is OPEN, which CLOSED |

### Limits — required because this task is contact-rich
| key | what to measure |
|---|---|
| `limits.force_torque_limits` | load at which motion must abort |
| `limits.max_speed` / `limits.max_acceleration` | low ceilings for this task |
| `limits.max_step_displacement` | largest displacement per supervised step |

### Supervision
| key | what to decide |
|---|---|
| `supervision.observation_freshness_s` | max observation age |
| `supervision.heartbeat_timeout_s` | supervisor heartbeat timeout |
| `supervision.success_predicate` | **measurable** predicate for success |
| `stop_conditions.commanded_observed_tolerance_deg` | allowed commanded-vs-measured drift |

Leaving the last one unset does **not** silently pass: disagreement becomes
`unevaluable` and the loop performs a controlled stop. That is tested.

### Only for `astra_direct`
`astra_direct.authorised_by`, `bounded_action_space`, `dry_run_passed`,
`collision_model`.

### Also required
`checkpoint.sha256` — digest of the exact pi0.5 checkpoint whose proposals are
reviewed.

## Data contract

Recorded episodes are described by `episode_manifest.schema.json`. **Media are
referenced by relative path + sha256 and are never committed to this repository.**

```
<episode_media_root>/                 # OUTSIDE this repo
  <episode_id>/
    manifest.json                     # validates against episode_manifest.schema.json
    media/base.mp4  media/wrist.mp4   # observation videos
    media/state.parquet               # synchronized robot state, one row per tick
    media/trajectory.parquet          # original recorded commanded trajectory
```

Alignment is enforced, not assumed: each video's `first_frame_epoch` must be
within 50 ms of `state.first_row_epoch`, and video duration must match state
duration. A skewed episode is rejected — reviewing frame *k* against a state row
from a different instant reviews nothing.

## Ports and endpoints

| what | where | notes |
|---|---|---|
| RSI UDP | Jetson `172.17.255.2:59152` | `KUKA_LOCAL_IP`; controller connects in |
| RSI rate | 250 Hz | `<Sen>` reply **must echo the received `IPOC`** |
| control rate | 30 Hz | chunk rate; the gateway interpolates up to RSI rate |
| Jetson web UI | `:8080` | LAN `192.168.3.7`, Tailscale `100.110.61.50` |
| A800 | proposals only | no route to the robot network |

`rsi_response_xml()` in `transports.py` shows the exact `<Sen>` frame shape for
cross-checking against `teleoperation/trigger_RSI/*.src`. Nothing in this package
sends it.

## Process ownership

| process | host | owner |
|---|---|---|
| pi0.5 inference server | A800 | ML |
| Astra review | eng-1 (offline) or A800 (gated) | ML |
| review/sanitize/validate loop | Jetson | deployment |
| RSI gateway | Jetson | deployment |
| supervisor heartbeat + deadman | Jetson-attached | **operator** |
| E-stop | physical interlock | **operator** |

The E-stop is hardware. Software reads its state and refuses; software can never
clear or bypass it. No code path writes E-stop state.

## Launch order

1. Verify the robot is in a safe pose, E-stop within reach.
2. Start the RSI gateway on the Jetson (shadow first).
3. Start the pi0.5 proposal server on the A800; confirm it is proposal-only.
4. Start the loop in `replay` and confirm audit rows appear.
5. Only then move to `live_shadow`.

## Health checks

```bash
# KUKA extension, offline, no robot
./.venv/bin/python -m pytest hybrid_rollout/robodojo/kuka -q
# whole repo including upstream
./.venv/bin/python -m pytest hybrid_rollout/robodojo -q
# confirm the shipped config still refuses to arm
./.venv/bin/python -c "from hybrid_rollout.robodojo.kuka.experiment import *; \
from hybrid_rollout.robodojo.kuka.safety import missing_config, Mode; \
print(missing_config(flatten_config(load_config()), Mode.REVIEWED_EXECUTION))"
```

Expected: a non-empty list. An empty list means someone filled in values — check
they were measured.

## Log locations

| what | where |
|---|---|
| cycle audit (all four modes, same shape) | path given to `AuditLog`, append-only JSONL |
| shadow "would send" records | `ShadowGateway.emitted` + the audit row |
| envelope digests | `envelope_sha256` per row; signatures are **never** logged |

## Run sheet

| phase | mode | precondition | stop when |
|---|---|---|---|
| 1 | `replay` | recorded episodes with validated manifests | review quality agreed |
| 2 | `live_shadow` | cameras live, robot idle | freshness + stop conditions behave |
| 3 | `reviewed_execution` | all 16 values measured, operator on deadman | any stop condition |
| 4 | `astra_direct` | phases 1–3 clean + 4 extra prerequisites | any stop condition |

---

## Group-chat message (copy; not sent)

> KUKA Astra review gateway is ready for offline review on branch
> `kuka-astra-review` (fork of GPT-as-Policy @ `8f3d362`). It runs four modes on
> one state machine: replay, live-shadow, reviewed-execution, astra-direct.
> Replay and shadow **cannot** emit a robot command — that's structural, not a
> flag. Upstream's gate prompt and decision schema are untouched; `eef` parses
> but is refused at the KUKA gate until TCP/IK/workspace/collision are verified.
>
> Before anyone can arm anything I need **22 measured values** on the real cell:
> handle pose, hinge axis, door-open region, table workspace, camera
> intrinsics/extrinsics, verified $TOOL/TCP, base frame, gripper polarity, F/T or
> controller load limits, max speed/accel, max per-step displacement, observation
> freshness, heartbeat timeout, a measurable success predicate, the
> commanded-vs-measured tolerance, and the checkpoint sha256. They're empty in
> the config on purpose — I'm not inventing numbers for a contact-rich door task.
>
> 562 offline tests pass (496 upstream + 66 new). No robot, simulator, API or GPU
> call has been made. Suggested next step: fill the config, run phase 1 (replay)
> on recorded episodes, then phase 2 (shadow) with cameras live and the arm idle.
