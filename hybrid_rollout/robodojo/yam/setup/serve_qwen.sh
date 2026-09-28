#!/usr/bin/env bash
# Qwen3-VL-2B monitor, OpenAI-compatible, on loopback:8020 -- the same model and
# endpoint shape as the KUKA branch (kuka/vlm_backends.SERVER_EXAMPLES).
# Needs vLLM in its own environment: uv venv ~/.venv-vllm && uv pip install --python ~/.venv-vllm/bin/python vllm
set -euo pipefail
VLLM="${VLLM:-$HOME/.venv-vllm/bin/vllm}"
exec "$VLLM" serve Qwen/Qwen3-VL-2B-Instruct --host 127.0.0.1 --port 8020 \
  --max-model-len 4096 --limit-mm-per-prompt '{"image": 4}' \
  --gpu-memory-utilization "${QWEN_GPU_FRACTION:-0.25}"
