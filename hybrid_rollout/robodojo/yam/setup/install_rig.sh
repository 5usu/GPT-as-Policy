#!/usr/bin/env bash
# Rig environment: .venv (git-ignored) at the repo root, Python 3.11, i2rt pinned.
# Safe to re-run. Touches no hardware.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
# i2rt builds ruckig from source; it needs a compiler and kernel headers for CAN.
for pkg in build-essential python3-dev; do
  dpkg -s "$pkg" >/dev/null 2>&1 || echo "WARNING: $pkg missing -- sudo apt install build-essential python3-dev linux-headers-\$(uname -r)"
done
[ -d .venv ] || uv venv --python 3.11 .venv
# ruckig 0.15.3 (pinned by i2rt) is sdist-only and fails to build under
# scikit-build-core >= 0.10; i2rt pins the same constraint for itself.
uv pip install --python .venv/bin/python \
  --build-constraints hybrid_rollout/robodojo/yam/setup/build-constraints.txt \
  -r hybrid_rollout/robodojo/yam/setup/requirements-rig.txt
.venv/bin/python -m pytest hybrid_rollout/robodojo/yam -q -p no:cacheprovider
echo
echo "rig environment ready: $ROOT/.venv"
echo "next: hybrid_rollout/robodojo/yam/setup/can_up.sh, then yam.cli doctor"
