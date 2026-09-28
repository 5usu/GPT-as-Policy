# YAM box setup runbook — for Claude Code, run on the YAM box

This takes a fresh clone of branch `yam-local-vlm-gate` to π0.5
(`robocurve/pi0.5-yam`) driving both YAM arms, with the Qwen3-VL-2B monitor and
Astra in the loop. Work through the steps **in order**. Every step says what it
touches. Record each measured value with `yam.cli set`; it goes into the
git-ignored `experiments/bimanual_blocks/rig.local.toml`.

## Rules for the agent

- **Never command the arms yourself.** Steps marked **OPERATOR** enable motors
  or move the arms. They refuse without `--hold`/`--execute`, and the moving
  ones also need the operator to type the rig id at an interactive terminal.
  Hand the exact command to the operator (they can run it with `! <command>`)
  and wait for the result.
- **Never invent a measured value.** If a value has to be measured or decided
  (rest pose, speed, tolerance, E-stop test, camera identity), ask the
  operator or get it from the step that measures it. Do not copy numbers from
  this file, from other rigs, or from the tests.
- `sudo` steps (CAN) need the operator's password. Ask; don't work around it.
- If a step fails, stop and report the output. Don't skip ahead.
- Nothing is sent to Astra (a paid API) without `--astra-live`, and that needs
  the operator's OK plus `OPENAI_API_KEY` set in their shell.

Shorthand used below: `Y` is `.venv/bin/python -m hybrid_rollout.robodojo.yam.cli`,
run from the repo root.

## 0. Look at the machine (read-only)

```bash
uname -a; python3 --version; nvidia-smi || cat /etc/nv_tegra_release
ip -br link | grep -i can; ls /dev/video*; df -h ~; free -g
```

Decide where each process runs and tell the operator:

| process | needs | where |
|---|---|---|
| rig loop (`yam.cli`) | CAN + cameras | this YAM box, always |
| π0.5 server | NVIDIA GPU, ≥16 GB, ~15 GB disk | here if there is such a GPU, otherwise a GPU host on the LAN |
| Qwen monitor | GPU, ~3 GB | here or the Jetson the KUKA branch used |

If the GPU is shared between π0.5 and Qwen, the scripts already split it:
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` for π0.5 and `QWEN_GPU_FRACTION=0.25` for Qwen.

## 1. Rig environment (no hardware)

```bash
hybrid_rollout/robodojo/yam/setup/install_rig.sh
```

This creates `.venv` (Python 3.11, i2rt pinned to `120c3c81`) and ends by
running the offline test suite, which must pass. If `uv` is missing, install
it first (`curl -LsSf https://astral.sh/uv/install.sh | sh`). If the
build-essential/python3-dev warning appears, ask the operator to install them
before continuing.

## 2. CAN (sudo; configures the interfaces, moves nothing)

```bash
hybrid_rollout/robodojo/yam/setup/can_up.sh
```

Ask the operator which interface is the LEFT arm and which is the RIGHT arm.
If the names aren't stable across replugs, point them to i2rt's
`docs/guides/set-persistent-can-ids.md`. Then:

```bash
$Y set rig.rig_id '"<a name the operator picks for this rig>"'
$Y set rig.left_can '"<left iface>"'
$Y set rig.right_can '"<right iface>"'
```

## 3. Cameras (read-only)

```bash
$Y cameras          # writes one JPEG per capturing /dev/video node to yam/runs/cameras/
```

**Open each image with the Read tool and look at it.** Top is the scene view;
left and right are the wrist cameras, and they sit on the matching arm. Tell
the operator what you see and get their confirmation. A swapped left/right
pair gives confident actions for the wrong scene. Then:

```bash
$Y set rig.camera_mapping '{"top": N, "left": N, "right": N}'
```

The frames must be 16:9 (640x360 is requested). If a camera can't deliver
16:9, report it; don't work around it.

## 4. Grippers — OPERATOR, the grippers move

```bash
$Y calibrate-grippers --execute
```

This runs i2rt's own sweep: each gripper closes and opens to its hard stops,
and the result is saved. **After it, the motors are off and the arms are
limp**, so the operator supports them or leaves them resting.

## 5. Poses — OPERATOR, motors on in gravity compensation

```bash
$Y record-pose --name rest_pose --hold    # a pose the arms can safely go limp in
$Y record-pose --name start_pose --hold   # optional: where each run starts
```

The operator hand-guides both arms into place and presses Enter. Every run
ends by parking at `rest_pose` and then disabling the motors.

## 6. Limits and supervision — decided by the operator

Ask the operator for each value and set it. Suggest a conservative first
setting, but let them choose:

```bash
$Y set limits.max_speed 0.5                  # rad/s   (hard ceiling 2.2)
$Y set limits.max_acceleration 1.5           # rad/s^2 (hard ceiling 6)
$Y set limits.max_step_displacement 0.3      # rad a single prefix may travel
$Y set supervision.observation_freshness_s 0.25
$Y set rig.estop_tested '"<date> <who> pressed E-stop, both arms dropped power"'
```

`estop_tested` is a record of an E-stop test the operator actually did. Ask
them to do one if they haven't.

## 7. Tracking tolerance — OPERATOR, both wrists rotate ±0.1 rad

```bash
$Y tracking-test --execute --save
```

This measures how far the arms trail a slow reference and saves a deviation
tolerance of three times the worst value seen (at least 0.05 rad). During
`execute`, a deviation beyond it stops the run.

## 8. π0.5 server (GPU host)

```bash
hybrid_rollout/robodojo/yam/setup/install_openpi.sh   # openpi @ 15a9616a + checkpoint @ ee17bb36
hybrid_rollout/robodojo/yam/setup/serve_pi05.sh       # 127.0.0.1:18840
```

`install_openpi.sh` ends by running `test_yam_openpi.py` inside the openpi
environment. That checks the reconstructed config's camera mapping, padding and
normalisation, and it must pass (it skips everywhere else). Run the server in
a separate terminal (or with `run_in_background`). On a
different host, pass `0.0.0.0` as the host and use that machine's address
below. Then check the reconstruction offline **before any arm moves**:

```bash
.venv/bin/hf download allenai/19012026-block-13 --repo-type dataset --local-dir ~/data/block13
.venv/bin/python -m hybrid_rollout.robodojo.yam.lerobot_episode --dataset ~/data/block13 \
  --episode 0 --every 16 --limit 20 --out ~/data/yam_ep0
$Y run --episode ~/data/yam_ep0 --pi05-url http://127.0.0.1:18840/infer
```

Also run `make_chunks` from the openpi environment for the per-tick error
against the recorded actions:

```bash
cd ~/openpi && PYTHONPATH=<repo> uv run python -m hybrid_rollout.robodojo.yam.make_chunks \
  --checkpoint ~/checkpoints/pi0.5-yam --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 \
  --episode ~/data/yam_ep0
```

A mean joint RMS well under 0.1 rad means the wiring is right. Large errors
mean a camera order, gripper convention or normalisation problem: **stop and
report**, and don't continue to the arms.

## 9. Qwen monitor (optional for the first run)

```bash
uv venv ~/.venv-vllm && uv pip install --python ~/.venv-vllm/bin/python vllm
hybrid_rollout/robodojo/yam/setup/serve_qwen.sh      # 127.0.0.1:8020
```

## 10. Check everything (read-only)

```bash
$Y doctor --pi05-url http://127.0.0.1:18840/infer --monitor-url http://127.0.0.1:8020/v1/chat/completions
```

Every line must be PASS; WARN lines are acceptable only if the operator agrees.

## 11. Observe with the arms holding — OPERATOR

```bash
$Y live --hold --pi05-url http://127.0.0.1:18840/infer \
  --monitor-url http://127.0.0.1:8020/v1/chat/completions
```

This runs the full three-model loop on the real scene while the arms hold.
Check the summary: π0.5 cycle time, `max_state_age_s`, no errors.

## 12. Run π0.5 on the arms — OPERATOR

```bash
$Y execute --execute --pi05-url http://127.0.0.1:18840/infer \
  --task "<prompt in the training data's phrasing, e.g. 'spell out NEURIPS'>" \
  --steps 8 --max-seconds 60 --policy-mode pi05_only
```

- **Stop:** Ctrl-C, `touch /tmp/yam_stop` from another terminal, or the E-stop.
  On a normal stop the arms park at `rest_pose`, then the motors go off. After a
  deviation fault the operator is asked before any parking move.
- **First run:** `pi05_only`, short `--max-seconds`, conservative limits.
- **With the monitor:** add `--policy-mode pi05_local_monitor_astra
  --monitor-url ...`. It is shadow by default and records only.
  `--monitor-gate` lets Qwen (and Astra on escalation) set the prefix length;
  a stale or missing decision then holds the arms.
- **With Astra:** add `--astra-live` (PAID) with `OPENAI_API_KEY` set.
- Audits go to `hybrid_rollout/robodojo/yam/runs/` (git-ignored).

## What to report back

The `doctor` output, the open-loop error from step 8, the `live` summary, and
each `execute` summary with its audit path.
