"""Bimanual I2RT YAM extension to the RoboDojo hybrid-rollout flow.

The KUKA branch's architecture -- pi0.5 proposes, a local Qwen3-VL-2B monitor
triages, Astra reviews on escalation, deterministic sanitize/validate/arm
gates decide what may leave -- on two YAM arms, with the public
robocurve/pi0.5-yam checkpoint in place of the KUKA full fine-tune.

Everything robot-independent is imported from `..kuka` unchanged. This package
adds only what knows the shape of a YAM row or talks to YAM hardware.

REAL MOTION IS IMPOSSIBLE BY DEFAULT. No code path in this package commands
the arms; `hold`/`live` connect to them read-only while they hold position.
"""
