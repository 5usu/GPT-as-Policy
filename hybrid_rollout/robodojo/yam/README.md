# Bimanual I2RT YAM extension

The `kuka-local-vlm-gate` architecture on two YAM arms. The policy is the public
[`robocurve/pi0.5-yam`](https://huggingface.co/robocurve/pi0.5-yam) checkpoint
instead of the KUKA full fine-tune; the Qwen monitor and Astra are unchanged.

```
cameras (top, left, right) + both arms' joints
  -> pi0.5 (robocurve/pi0.5-yam, 16x14 chunk)       yam/pi05_serve.py, GPU host
  -> Qwen3-VL-2B monitor on the Jetson (triage)      kuka/vlm_backends.py, unchanged
  -> Astra, only on persistent adverse evidence      kuka/transports.py, unchanged
  -> deterministic sanitize -> YAM validation -> arming gates
  -> motion.py (Ruckig, 100 Hz) -> both arms     `yam.cli execute`, operator-armed
```

## Relation to the KUKA branch

Branched from `kuka-local-vlm-gate` @ `4acf6d2`. **No existing file is
modified**: not upstream, not `kuka/`, not `requirements-tools.txt`. The only
file added outside this directory is the root `CLAUDE.md`, which points an
agent on the YAM box at SETUP.md.

| reused from `kuka/`, unchanged | YAM-specific, here |
|---|---|
| Qwen monitor backend, prompt, response schema | `contract.py`: 14 dims, radians, 16-step chunks, gripper 0 = closed |
| `MonitorGate`: streaks, clamps, fail-safe, hand-back | `pipeline.py`: bimanual event detector, bimanual escalation packet |
| `PolicyMode` routing and aliases | `schema.py`: upstream's `{left, right}` wrappers kept, edit in joint radians |
| Astra client (background submit+poll, no re-ask) | `packet.py`: gate prompt verbatim + YAM contract note |
| modes, supervisor, freshness, signed envelope, ledger | `safety.py`: the values a YAM rig must measure; rig identity |
| audit row, stages, outcomes; manifest validation | `validation.py`, `sanitize.py`: the same rules over both arms |
| camera grabber and JPEG encoding | `cameras.py` (3 cameras, 16:9), `robot.py` (i2rt) |
| | `motion.py`, `execute.py`: the execution path (below) |

Upstream was already dual-arm (ARX-X5), so the decision schema is closer to
upstream here than on the KUKA: `left`/`right` stay.

## What differs from the KUKA cell, and why it matters

| | KUKA | YAM |
|---|---|---|
| gripper polarity | >= 0.5 closing | **0 = closed, 1 = open** |
| gripper state fed to pi0.5 | measured | **last commanded opening** (what the dataset logged) |
| units | degrees | radians |
| chunk | 50 x 7 | 16 x 14 |
| hold | answer every 4 ms RSI frame | i2rt holds the connect pose; **exit = motors off, arms limp** |
| gripper bus | separate Modbus | same CAN chain as its arm |
| camera frames | 640x480 | **640x360** (trained 16:9; 4:3 changes the padding) |
| pi0.5 contract check | `use_relative_actions=True` | repo, norm-stats sha256, 16x14, polarity |

## Getting the checkpoint running

`robocurve/pi0.5-yam` is openpi JAX (~12 GB of params). The model card loads it
with a `yam_pi05` config registered by a **private** training repo, so
`pi05_serve.py` rebuilds that config from what is published and checkable
(the card, the checkpoint's own norm stats, the dataset's feature layout, and
the vla-edge serving spec for the same checkpoint). **It is a reconstruction:
check it on a held-out recording before any arm is involved.**

```bash
# GPU host, openpi environment (openpi @ 15a9616a)
hf download robocurve/pi0.5-yam --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 \
    --local-dir /ckpt/pi0.5-yam

# 1. offline check against the session robocurve held out
hf download allenai/19012026-block-13 --repo-type dataset --local-dir /data/block13
python -m hybrid_rollout.robodojo.yam.lerobot_episode --dataset /data/block13 \
    --episode 0 --every 16 --limit 20 --out /data/yam_ep0          # needs pyarrow + ffmpeg
python -m hybrid_rollout.robodojo.yam.make_chunks --checkpoint /ckpt/pi0.5-yam \
    --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 --episode /data/yam_ep0
#    -> per-tick joint RMS vs the recorded actions; large errors = wiring, stop here

# 2. serve it (loopback by default; proposal-only) -- or setup/serve_pi05.sh
python -m hybrid_rollout.robodojo.yam.pi05_serve --checkpoint /ckpt/pi0.5-yam \
    --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 --port 18840
```

The Jetson in `kuka/vlm_backends.JETSON_COMPUTE` is an AGX **Orin**. The
published TensorRT bundle for this checkpoint
(`agents2agents/Pi0.5-BimanualYAM-Jetson-Thor`) needs a **Thor**, so on this
Jetson pi0.5 runs on a separate GPU host and Qwen stays on the Jetson.

## Stages (none skippable, same as the KUKA run sheet)

```bash
Y="python -m hybrid_rollout.robodojo.yam.cli"
$Y preflight --pi05-url http://127.0.0.1:18840/infer          # contract + missing values
$Y phase1 --episode /data/yam_ep0                              # Astra packets, nothing sent
$Y run --episode /data/yam_ep0 --pi05-url http://127.0.0.1:18840/infer \
       --policy-mode pi05_local_monitor_astra --monitor-backend local \
       --monitor-url http://127.0.0.1:8020/v1/chat/completions  # add --astra-live to pay
$Y hold --hold --left-can can_follower_l --right-can can_follower_r \
        --gripper-limits '{"left": [c, o], "right": [c, o]}'   # arms hold, 10 s, read-only
$Y live --hold --pi05-url http://.../infer --cameras top:N,left:N,right:N \
        --monitor-url http://127.0.0.1:8020/v1/chat/completions  # full loop, arms hold
$Y execute --execute --pi05-url http://.../infer --task "..."    # pi0.5 moves the arms
```

`hold` and `live` **connect to the arms, which enables the motors**. They hold
the pose they are in and command nothing -- `robot.HeldArms` has no command
method -- but on exit i2rt disables the motors and **the arms go limp**. Start
from and end in a pose they can rest in, E-stop in reach. Gripper calibration
limits are required so that connecting never runs i2rt's gripper sweep.

## Execution: where this branch goes further than the KUKA one

The KUKA branch never gave the policy a motion path. Here, `yam.cli execute`
runs pi0.5 on the arms, the way the published YAM pi0.5 controller runs this
checkpoint:

- `motion.py` is the **only** caller of `command_joint_pos`: a jerk-limited
  Ruckig reference at 100 Hz, fed with 30 Hz chunk rows. A measured-vs-reference
  deviation above the measured tolerance brakes and holds.
- Every chunk goes through validate -> (prefix) -> sanitize -> re-validate ->
  displacement cap before any row becomes a goal. FATAL violations, server
  errors and stale frames HOLD.
- The Qwen monitor runs **asynchronously**, because a 5-9 s reading can't sit
  inside a 30 Hz cycle. It is shadow by default. With `--monitor-gate` the
  prefix length is the monitor's (or Astra's) clamped number, and a missing or
  stale decision means 0 steps: hold. An Astra `stop` ends the run.
- It refuses to arm until every value in `execute.EXECUTE_REQUIRED` is measured
  (SETUP.md shows how), the CAN links are up and the pi0.5 server passes the
  contract check. Then an operator must type the rig id at the terminal.
- It stops on Ctrl-C, `touch /tmp/yam_stop` or the E-stop, parks at
  `rest_pose` at 0.3 rad/s, and only then disables the motors.

Still locked, as on the KUKA branch: reviewer `edit`s are computed but not
applied, `eef` is refused, Astra-direct isn't implemented, and the review
loop's gateway is `ShadowGateway`.

**Setting up a rig: follow [SETUP.md](SETUP.md).**

## Offline tests

```bash
./.venv/bin/python -m pytest hybrid_rollout/robodojo/yam -q
```

No robot, camera, model, GPU or paid call. `test_yam_openpi.py` runs the
reconstructed openpi transform chain when openpi is importable and skips
otherwise.
