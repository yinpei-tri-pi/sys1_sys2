#!/usr/bin/env bash
# One-time setup on a new machine (run from anywhere; ~40 GB disk, mostly the System2 image + assets):
#   1. git submodules: official RoboCasa 1.0.1 + robosuite 1.5.2
#   2. System1 env    system1/.venv  (uv, pinned by system1/uv.lock)
#   3. System2 image  sys1-sys2/system2-vllm:0.19.1  (Docker, vLLM 0.19.1)
#   4. simulator env  sim/.venv  + RoboCasa kitchen assets (~10 GB download)
# Skip a step with SKIP_SYSTEM1=1 / SKIP_SYSTEM2=1 / SKIP_SIM=1. Needs git, uv, Docker + NVIDIA toolkit.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
command -v uv >/dev/null || { echo "FATAL: uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

git submodule update --init third_party/robocasa third_party/robosuite
if [[ "${SKIP_SYSTEM1:-0}" != 1 ]]; then
  echo "[setup] System1 env"; (cd system1 && uv sync --frozen)
fi
if [[ "${SKIP_SYSTEM2:-0}" != 1 ]]; then
  echo "[setup] System2 image"; docker build -t sys1-sys2/system2-vllm:0.19.1 system2
fi
if [[ "${SKIP_SIM:-0}" != 1 ]]; then
  echo "[setup] simulator env + kitchen assets"; bash scripts/setup_sim_env.sh
fi
echo "[setup] done. Next: bash scripts/download_checkpoints.sh"
