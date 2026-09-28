#!/usr/bin/env bash
# pi0.5 host environment: openpi at the commit robocurve trained with, plus the
# checkpoint. Needs an NVIDIA GPU with >= 16 GB (bf16 pi0.5, ~12 GB params on disk).
# Usage: install_openpi.sh [OPENPI_DIR] [CKPT_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
OPENPI_DIR="${1:-$HOME/openpi}"
CKPT_DIR="${2:-$HOME/checkpoints/pi0.5-yam}"
OPENPI_COMMIT=15a9616a00943ada6c20a0f158e3adb39df2ccac
CKPT_REV=ee17bb361e95eeba57853a2840480f5a1fc81a84
command -v uv >/dev/null || { echo "install uv first"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || \
  echo "WARNING: no nvidia-smi; openpi will fall back to CPU, far too slow for control"
if [ ! -d "$OPENPI_DIR/.git" ]; then
  git clone https://github.com/Physical-Intelligence/openpi "$OPENPI_DIR"
fi
cd "$OPENPI_DIR"
git fetch -q origin && git checkout -q "$OPENPI_COMMIT"
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install "huggingface_hub[cli]"
uv run hf download robocurve/pi0.5-yam --revision "$CKPT_REV" --local-dir "$CKPT_DIR"
# The reconstructed openpi config, checked inside the real openpi env (no weights).
PYTHONPATH="$ROOT" uv run --with pytest python -m pytest -q -p no:cacheprovider \
  "$ROOT/hybrid_rollout/robodojo/yam/test_yam_openpi.py"
echo
echo "openpi: $OPENPI_DIR   checkpoint: $CKPT_DIR"
echo "next: hybrid_rollout/robodojo/yam/setup/serve_pi05.sh $OPENPI_DIR $CKPT_DIR"
