# GPT-as-Policy — branch `yam-local-vlm-gate`

This branch adds `hybrid_rollout/robodojo/yam/`. That package runs
`robocurve/pi0.5-yam` on two I2RT YAM arms, with the Qwen3-VL-2B monitor and
the Astra reviewer from the `kuka/` package.

- **Setting up a YAM box, or asked to "set things up" / "run on the YAMs":**
  follow `hybrid_rollout/robodojo/yam/SETUP.md` step by step. It says which
  steps move hardware; hand those to the operator and never run them yourself.
- Architecture and what is reused from `kuka/`: `hybrid_rollout/robodojo/yam/README.md`.
- Offline tests (no hardware, model or network):
  `.venv/bin/python -m pytest hybrid_rollout/robodojo/yam -q`
- Don't modify upstream files or `hybrid_rollout/robodojo/kuka/`. YAM changes
  belong in `hybrid_rollout/robodojo/yam/`.
- Measured rig values go in the git-ignored `yam/experiments/*/rig.local.toml`
  via `yam.cli set`. Never commit them, and never put them in `config.toml`.
