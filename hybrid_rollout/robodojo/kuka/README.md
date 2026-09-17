# KUKA LBR iisy 11 R1300 extension

A KUKA-specific extension of the upstream RoboDojo hybrid-rollout flow. It is
**not** a parallel application: it adapts to the upstream interfaces and reuses
the upstream gate verbatim.

## Relation to upstream

| | |
|---|---|
| upstream | https://github.com/anonymous-report-421/GPT-as-Policy |
| commit | `8f3d362b077d8efb77e2a7274d5b2c20e2243846` ("Initial public release") |
| licence | MIT © 2026 Yu-Mool Shu and Lipxin Zheng — preserved, unmodified |
| branch | `kuka-astra-review` |

**Nothing upstream is modified.** `skill/gate_prompt.md` is byte-identical and its
sha256 still matches `robodojo/SOURCE.json` (`5bb54e0e…`). This extension adds one
directory and touches nothing outside it.

### What is reused, not reimplemented
- `skill/gate_prompt.md` — the outcome/intent gate, verbatim
- `skill/schema.py` — decision contract shape (`student`/`edit`/`eef`/`stop`, the
  assessment block, step bounds 1–15 / 1–5)
- `robodojo_server/kinematics.py` `ArmFK` — URDF chain FK, used directly

### What differs, and why
| upstream | here | forced by |
|---|---|---|
| dual ARX-X5, `{left,right}` wrappers | one 6-DoF arm, wrappers dropped | embodiment |
| `edit` in Cartesian m / rad | `edit` in **joint-space degrees** | no verified TCP, no IK |
| simulator RPC command path | **Jetson RSI UDP/XML** gateway | real hardware |
| `eef` executable | `eef` **parsed and refused** at the execution gate | see below |

`eef` stays in the schema and the parser because it is a legitimate upstream
decision and removing it would fork the contract. It is refused at the KUKA
execution gate until all four of `tcp_transform_verified`, `ik_implemented`,
`workspace_configured` and `collision_configured` are true. It is accepted,
recorded, and never becomes a command.

## The preserved review path

```
observation -> pi0.5 proposal -> FK preview -> Astra review
  -> student/edit/eef decision -> deterministic sanitize -> KUKA validation
  -> signed command envelope -> Jetson RSI gateway -> feedback -> replan
```

FK is **not** a simulation of contact, grasping or success (upstream's wording,
kept). It maps commanded joint targets to a flange pose. The custom `$TOOL`
transform is unverified, so these are not tool-tip positions.

## Roles

| host | role | may command the robot |
|---|---|---|
| **eng-1** | development, offline analysis. Holds the current `OPENAI_API_KEY`, which is **not** copied anywhere. | no |
| **A800** | pi0.5 inference → proposed action chunks. **Proposal-only by default.** | no |
| **Jetson** | the **only** KUKA command gateway. RSI UDP/XML at 250 Hz. | yes, once armed |

Live Astra review from the A800 needs **both** `A800_LIVE_REVIEW=1` **and** a new
dedicated `A800_ASTRA_API_KEY`. The eng-1 key must not be copied to the A800.

## Four modes, one state machine, one audit format

| mode | proposals from | reaches | can command |
|---|---|---|---|
| `replay` | recorded trajectory | `validate` | **no** — structurally |
| `live_shadow` | live cameras | `validate` | **no** — structurally |
| `reviewed_execution` | pi0.5 | `replan` | one supervised step |
| `astra_direct` | Astra itself | `replan` | one step, +4 prerequisites |

`replay` and `live_shadow` cannot construct a command object at all —
`require_command_capable` raises before anything is built. That is structural, not
a configuration flag someone can flip.

## Offline setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements-tools.txt
./.venv/bin/python -m pytest hybrid_rollout/robodojo -q        # upstream + KUKA
```

The KUKA extension itself is **stdlib-only** except for the optional FK preview
(numpy/scipy). It runs unchanged on a Jetson without the analysis stack.

## Shadow mode

`ShadowGateway` has **no socket**. Shadow mode is the absence of a transport, not
a disabled one: the envelope is built, signed and validated, then logged with
`sent: false`.

## Staged rollout

1. **replay** — recorded episodes only. Confirm review quality and audit shape.
2. **live_shadow** — live cameras, commands suppressed. Confirm freshness,
   stop conditions and success predicate against reality.
3. **reviewed_execution** — one supervised step, human on the deadman. Requires
   all 16 config values measured.
4. **astra_direct** — only after 1–3 are clean and the extra four prerequisites
   exist. Intended to be hard to arm.

Nothing may skip a stage. See `DEPLOYMENT.md` for the run sheet.
