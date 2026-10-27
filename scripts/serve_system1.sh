#!/usr/bin/env bash
# Serve ONE System1 (pi0.5) checkpoint on ONE GPU over the openpi websocket protocol. Blocks.
#
#   bash scripts/serve_system1.sh --ckpt <run>/269999 --gpu 0 --port 8060 [--mem-frac 0.35]
#
# --ckpt     checkpoint STEP dir (params/ + assets/robocasa_system1/norm_stats.json); the run root's
#            config.json selects the model/prompt configuration.
# --mem-frac JAX's share of the GPU. Leave room when the simulator (EGL) and/or System2 share the card.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CKPT="" GPU=0 PORT=8060 MEM_FRAC=0.35
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --mem-frac) MEM_FRAC=$2; shift 2 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
source "$ROOT/scripts/checkpoints.env"
CKPT=${CKPT:-$ROOT/$S1_CKPT_DEFAULT}   # default: the downloaded checkpoint
[[ -d "$CKPT" ]] || { echo "FATAL: no checkpoint at $CKPT (pass --ckpt or run scripts/download_checkpoints.sh)" >&2; exit 2; }
CKPT=$(cd "$CKPT" && pwd)
[[ -d "$CKPT/params" ]] || { echo "FATAL: $CKPT has no params/ (pass the step dir, e.g. .../269999)" >&2; exit 1; }
PY="$ROOT/system1/.venv/bin/python"
[[ -x "$PY" ]] || { echo "FATAL: $PY missing; run: (cd system1 && uv sync --frozen)" >&2; exit 1; }

echo "[system1] ckpt=$CKPT gpu=$GPU port=$PORT mem_frac=$MEM_FRAC"
cd "$ROOT/system1"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
  exec "$PY" serve_system1.py --ckpt "$CKPT" --port "$PORT"
