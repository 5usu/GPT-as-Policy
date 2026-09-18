# Deployment run sheet — KUKA dishwasher cell

Every value below must be **measured on the real cell** and supplied by the
deployment engineer. They are empty on purpose. Arming refuses and names each one
until it is filled, and a plausible default would be worse than no value because
it would look like knowledge.

**Nothing in this repository has commanded the arm. Execution is disabled by
default and all Cartesian modes are locked by the interface itself.**

---

## 0. Verified cell facts (already encoded — no action needed)

| fact | value |
|---|---|
| robot NIC | `eno1` `172.17.255.2/16` → KUKA KLI `172.17.255.1`, ~0.79 ms, no loss |
| isolated from robot path | `wlp1s0`, outside-world NIC |
| RSI control | **UDP 59152** — controller is the client, Jetson answers every 4 ms |
| RSI reply | `<Sen Type="ImFree">` with `AK.A1..A6`, `STOPFLAG`, **identical IPOC** |
| RSI receive config accepts | `AK.A1..A6` + `STOPFLAG` — **no RKorr** |
| program trigger | **TCP 54600** (EthernetKRL/iicoServer) — *not* the control stream, currently **closed** |
| OPC UA | TCP 4840 — supervision only, **never** robot control |
| rate stack | π0.5 1–3 Hz, ACT 30 Hz → RSI 250 Hz via `Ruckig(NUM_JOINTS, 0.004)` |
| gripper | `/dev/ttyUSB0`, CH340 → Modbus RTU/RS485, **direct from Jetson, not via controller** |
| cameras | 2× Tera USB; nodes 0/1 and 2/3, only **0 and 2** capture |
| authority | **only the Jetson may command**; A800 proposes and has no route |

---

## 1. Measured values required before any motion (16)

### Cell geometry — robot base frame
- [ ] `handle_pose` — dishwasher door handle pose
- [ ] `door_hinge_axis` — hinge origin + direction
- [ ] `door_open_region` — expected door sweep; motion outside it stops the run
- [ ] `table_workspace` — allowed Cartesian volume

### Calibration
- [ ] `camera_intrinsics` — K + distortion, base and wrist
- [ ] `camera_extrinsics` — camera poses in base frame
- [ ] `tcp_transform` — **$TOOL read off the controller**, then verified against `RIst`
- [ ] `base_frame` — verified base frame definition
- [ ] `gripper_polarity` — **which commanded value is OPEN, which is CLOSED**

### Limits — required because this task is contact-rich
- [ ] `force_torque_limits` — load at which motion must abort
- [ ] `max_speed`, `max_acceleration` — low ceilings for this task
- [ ] `max_step_displacement` — largest displacement per supervised step

### Supervision
- [ ] `observation_freshness_s` — maximum observation age
- [ ] `heartbeat_timeout_s` — supervisor heartbeat timeout
- [ ] `success_predicate` — a **measurable** predicate, not model prose

### Also required
- [ ] `commanded_observed_tolerance_deg` — leaving this unset makes the
      commanded-vs-measured check **unevaluable**, which performs a controlled
      stop rather than passing. That is tested.
- [ ] camera mapping — `{"base": 0, "wrist": 2}` or the reverse. **The code
      refuses to guess**, and refuses a metadata node (1 or 3), which would open
      cleanly and yield nothing.

### For `astra_direct` only (4 more)
- [ ] `astra_direct_authorised_by`, `bounded_action_space`, `dry_run_passed`,
      `collision_model`

---

## 2. Phases, in order — none skippable

| phase | what | robot | gate |
|---|---|---|---|
| 1 | Astra reviews a recorded trajectory + videos | none | ready now |
| 2 | live camera shadow review | idle | camera mapping |
| 3 | HOLD on the real controller | powered, still | see §3 |
| 4 | supervised π0.5 + Astra, one step | moving | all 16 values |
| 5 | Astra-direct | moving | **blocked — see §4** |

---

## 3. HOLD checklist — the first real-robot step

HOLD answers every 4 ms frame with the pose the arm is already at. It keeps a
session alive and commands nothing.

- [ ] arm powered, E-stop within reach
- [ ] start the controller-side RSI program (TCP 54600 trigger) — currently closed
- [ ] `preflight` shows `ready for HOLD: True`
- [ ] run HOLD for a sustained period and confirm:
  - [ ] **zero late replies**
  - [ ] **no IPOC regressions**
  - [ ] no controller fault

**A controlled stop keeps replying with `Stopflag=1`. Silence faults the
controller, so going quiet is a failure mode, not a safe state.**

**The hardware E-stop is authoritative. Software may observe and refuse; it never
clears or bypasses it. No code path here writes E-stop state.**

---

## 4. What is locked, and why

| mode | status | reason |
|---|---|---|
| `student` prefix | **the only executable mode** | bounded prefix of π0.5 joint actions |
| `edit` | locked | deliverable over `AK.A1..A6`, but disabled pending a supervised campaign. The edit is still computed, validated and recorded — just not emitted |
| `eef` | **locked by the interface** | the RSI receive config has **no RKorr**, so a Cartesian correction cannot be delivered at all |
| `astra_direct` | **locked by the interface** | Cartesian-only upstream, so it inherits the same blocker |

**Supplying the $TOOL transform alone would not unlock `eef`.** Two independent
blockers must clear: the RSI configuration must gain `RKorr`, *and* the Tool
Center Point must be calibrated and verified.

---

## 5. Interpolation

The cell fills 250 Hz with `Ruckig(NUM_JOINTS, 0.004)` — jerk-limited. This
package ships a **linear** bridge that is explicitly marked **not deployable**:
it is velocity-discontinuous at segment boundaries. `RSIGateway` refuses
`enable_motion=True` unless a deployable interpolator is supplied, and does not
substitute linear for real motion.

The interpolated trajectory is validated **at every cycle** against position and
velocity limits — a chunk feasible on average can hide one infeasible 4 ms step,
and that step is what the controller sees.

---

## 6. Gripper — a separate actuator

It does **not** pass through the KUKA controller, so:

- a controlled stop on the RSI loop stops the **arm**. The gripper holds its last
  commanded state. **If it was closing on the handle, it stays closed.**
- RSI faults on a late reply; Modbus does not, so a wedged gripper write produces
  no controller-side symptom
- `gripper_polarity` is unknown for this cell, and getting it backwards means
  gripping when you meant to release — on a door handle

Any stop procedure must decide **explicitly** what the gripper does, and record it.

---

## 7. Still absent

- the CLI-to-RSI output path is not wired — the Astra loop reads recorded frames
  and writes audit output; the live camera layer is only the **input** side
- the 16 measured values
- the controller-side RSI program is not running
- the Jetson RSI socket is unbound

**Do not claim real-arm readiness until §3 passes and §1 is complete.**
