#!/usr/bin/env bash
# Serve robocurve/pi0.5-yam on loopback:18840. Proposal-only; no robot handle.
# Usage: serve_pi05.sh [OPENPI_DIR] [CKPT_DIR] [HOST] [PORT]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
OPENPI_DIR="${1:-$HOME/openpi}"
CKPT_DIR="${2:-$HOME/checkpoints/pi0.5-yam}"
HOST="${3:-127.0.0.1}"
PORT="${4:-18840}"
# Leave GPU memory for the Qwen monitor when both share one GPU.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
cd "$OPENPI_DIR"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec uv run python -m hybrid_rollout.robodojo.yam.pi05_serve \
  --checkpoint "$CKPT_DIR" --revision ee17bb361e95eeba57853a2840480f5a1fc81a84 \
  --host "$HOST" --port "$PORT"
