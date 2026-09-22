#!/usr/bin/env bash
# Astra review of pi0.5 proposals for one recorded episode. SHADOW ONLY:
# nothing here can move the robot -- live_shadow cannot construct a command.
#
#   ./review_episode.sh <episode_dir> [--live] [--limit N] [--every N]
#
# Without --live the reviewer is a DRY RUN: packets are built and hashed, no
# API call, no cost. With --live it calls gpt-6-astra (billable, ~3k tokens per
# tick). Step 1 (pi0.5) runs in a network-less container and is cached: delete
# runs/chunks_<episode>.json to force it again.
set -euo pipefail

REPO=/home/hexfellow/GPT-as-Policy
JET=/home/hexfellow/GPT-as-Policy-jetson
CKPT=/home/hexfellow/KUKA/teleoperation/checkpoints/pi05_corrected_b8/132000

EP=${1:?usage: review_episode.sh <episode_dir> [--live] [--limit N] [--every N]}
shift
EP=$(cd "$EP" && pwd)
NAME=$(basename "$EP")
LIVE=(); LIMIT=5; EVERY=50
while [ $# -gt 0 ]; do
  case "$1" in
    --live)  LIVE=(--astra-live --astra-effort low --astra-attempts 3); shift ;;
    --limit) LIMIT=$2; shift 2 ;;
    --every) EVERY=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

CHUNKS=$JET/runs/chunks_$NAME.json
mkdir -p "$JET/runs"

# --- 1. pi0.5 proposals, offline -------------------------------------------
if [ ! -s "$CHUNKS" ]; then
  echo "== pi0.5 on $NAME (loads the 9.35 GB checkpoint, ~6 min, no network)"
  docker run --rm --runtime nvidia --network none --name pi05-chunks \
    -v "$REPO":/workspace/GPT-as-Policy:ro \
    -v "$JET":/workspace/jetson \
    -v /home/hexfellow/KUKA/teleoperation:/workspace/KUKA/teleoperation:ro \
    -v "$CKPT":/ckpt:ro \
    -v /home/hexfellow/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_CACHE=/root/.cache/huggingface/hub -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 -e PYTHONIOENCODING=utf-8 -e LANG=C.UTF-8 \
    -w /workspace/GPT-as-Policy \
    pi05-infer:latest python3 -u /workspace/jetson/make_chunks.py \
      "/workspace/jetson/episodes/$NAME" "/workspace/jetson/runs/chunks_$NAME.json" \
      --every "$EVERY" --limit "$LIMIT" 2>&1 | grep -vE "^(WARNING|The PI05|This impl|Original impl| *observation\[)"
else
  echo "== reusing $CHUNKS"
fi

# --- 2. the gated review loop ----------------------------------------------
echo "== review loop (${LIVE[*]:-dry run})"
[ -n "${LIVE[*]:-}" ] && . /home/hexfellow/.config/gpt-as-policy/env.sh
cd "$REPO"
exec ./.venv/bin/python -u -m hybrid_rollout.robodojo.kuka.cli run \
  --manifest "$EP/manifest.json" --state-json "$EP/state.json" \
  --trajectory-json "$EP/trajectory.json" --media-root "$EP" \
  --chunks-file "$CHUNKS" --every "$EVERY" --limit "$LIMIT" \
  --audit "$JET/runs/audit_$NAME.jsonl" "${LIVE[@]}"
