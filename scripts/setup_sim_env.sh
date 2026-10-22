#!/usr/bin/env bash
# Build the simulator env (official RoboCasa 1.0.1 + robosuite 1.5.2 from the submodules) at sim/.venv.
#
#   bash scripts/setup_sim_env.sh            # env + kitchen assets (~10 GB download)
#   SKIP_ASSETS=1 bash scripts/setup_sim_env.sh
#
# Mirrors robocasa's README install (conda python=3.11 + `pip install -e robosuite` +
# `pip install -e robocasa` + setup_macros + download_kitchen_assets), using uv for the venv.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
UV=${UV:-uv}
VENV=$ROOT/sim/.venv
PY=$VENV/bin/python

git submodule update --init third_party/robocasa third_party/robosuite
[[ -x "$PY" ]] || "$UV" venv --python 3.11 "$VENV"
"$UV" pip install --python "$PY" -e third_party/robosuite -e third_party/robocasa imageio-ffmpeg
# openpi-client pins numpy<2 but works unchanged on robocasa's numpy 2.2.5; install it without its
# pins and add its runtime deps explicitly.
"$UV" pip install --python "$PY" --no-deps -e system1/packages/openpi-client
"$UV" pip install --python "$PY" dm-tree msgpack websockets

RC_PKG=$ROOT/third_party/robocasa/robocasa
# Both robocasa scripts prompt (y/n) when their target already exists, so only run them when needed.
[[ -f "$RC_PKG/macros_private.py" ]] || "$PY" -m robocasa.scripts.setup_macros
if [[ "${SKIP_ASSETS:-0}" != 1 && ! -d "$RC_PKG/models/assets/objects/objaverse" ]]; then
  yes | "$PY" -m robocasa.scripts.download_kitchen_assets   # answers its "Proceed? (y/n)" prompts
fi
MUJOCO_GL=egl "$PY" -c 'import robocasa, robosuite, openpi_client; print("robocasa", robocasa.__version__, "robosuite", robosuite.__version__)'
