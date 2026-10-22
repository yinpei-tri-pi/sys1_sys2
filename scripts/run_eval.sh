#!/usr/bin/env bash
# One evaluation stack: System1 server + System2 server + the official-protocol RoboCasa loop.
#
#   bash scripts/run_eval.sh --s1-ckpt <s1 step dir> --s2-ckpt <s2 checkpoint dir> \
#       --s1-gpu 0 --s2-gpu 1 --tasks atomic_seen --num-trials 50 --log-dir results/<run>
#
# --tasks       comma-separated task names and/or groups: atomic_seen,composite_seen,composite_unseen,all
# --sim-gpu     GPU for MuJoCo EGL rendering (default: the System1 GPU)
# --skip-servers  reuse servers already listening on --s1-port/--s2-port
# Anything after `--` is passed to sim/official_eval.py (e.g. -- --no-task-rules).
# Servers are stopped when the evaluation exits. Logs: <log-dir>/logs/.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

S1_CKPT="" S2_CKPT="" S1_GPU=0 S2_GPU=1 SIM_GPU="" S1_PORT=8060 S2_PORT=8100
S1_MEM=0.35 S2_MEM=0.45 TASKS="atomic_seen" NUM_TRIALS=50 SPLIT=pretrain SEED=7 LOG_DIR=""
SKIP_SERVERS=0 GPU_MODE=${DOCKER_GPU_MODE:-gpus} EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --s1-ckpt) S1_CKPT=$2; shift 2 ;;
    --s2-ckpt) S2_CKPT=$2; shift 2 ;;
    --s1-gpu) S1_GPU=$2; shift 2 ;;
    --s2-gpu) S2_GPU=$2; shift 2 ;;
    --sim-gpu) SIM_GPU=$2; shift 2 ;;
    --s1-port) S1_PORT=$2; shift 2 ;;
    --s2-port) S2_PORT=$2; shift 2 ;;
    --s1-mem-frac) S1_MEM=$2; shift 2 ;;
    --s2-mem-frac) S2_MEM=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --num-trials) NUM_TRIALS=$2; shift 2 ;;
    --split) SPLIT=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --skip-servers) SKIP_SERVERS=1; shift ;;
    --docker-gpu-mode) GPU_MODE=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$S1_CKPT" && -n "$S2_CKPT" && -n "$LOG_DIR" ]] \
  || { echo "FATAL: --s1-ckpt, --s2-ckpt and --log-dir are required" >&2; exit 2; }
S1_CKPT=$(cd "$S1_CKPT" && pwd); S2_CKPT=$(cd "$S2_CKPT" && pwd)
SIM_GPU=${SIM_GPU:-$S1_GPU}
mkdir -p "$LOG_DIR/logs"; LOG_DIR=$(cd "$LOG_DIR" && pwd)
SIM_PY="$ROOT/sim/.venv/bin/python"
[[ -x "$SIM_PY" ]] || { echo "FATAL: $SIM_PY missing; run scripts/setup_sim_env.sh" >&2; exit 1; }

pids=()
cleanup() {
  trap - EXIT INT TERM
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  [[ "$SKIP_SERVERS" == 1 ]] || docker stop "sys1sys2-system2-$S2_PORT" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$SKIP_SERVERS" != 1 ]]; then
  bash "$ROOT/scripts/serve_system1.sh" --ckpt "$S1_CKPT" --gpu "$S1_GPU" --port "$S1_PORT" \
    --mem-frac "$S1_MEM" > "$LOG_DIR/logs/system1.log" 2>&1 &
  pids+=("$!")
  bash "$ROOT/scripts/serve_system2.sh" --ckpt "$S2_CKPT" --gpu "$S2_GPU" --port "$S2_PORT" \
    --mem-frac "$S2_MEM" --docker-gpu-mode "$GPU_MODE" > "$LOG_DIR/logs/system2.log" 2>&1 &
  pids+=("$!")
  # A cold vLLM start (image import + torch.compile + CUDA graphs) can take ~20 min.
  echo "[run_eval] waiting for System1 :$S1_PORT and System2 :$S2_PORT (logs in $LOG_DIR/logs)"
  for _ in $(seq 1 240); do
    s1=$( (exec 3<>"/dev/tcp/127.0.0.1/$S1_PORT") 2>/dev/null && echo up || echo down)
    s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$S2_PORT/v1/models" || true)
    [[ "$s1" == up && "$s2" == 200 ]] && break
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "FATAL: a server exited; see $LOG_DIR/logs" >&2; exit 1; }; done
    sleep 10
  done
  [[ "$s1" == up && "$s2" == 200 ]] || { echo "FATAL: servers not ready after 40 min" >&2; exit 1; }
fi
echo "[run_eval] servers ready; evaluating tasks=[$TASKS] trials=$NUM_TRIALS split=$SPLIT seed=$SEED"

read -ra TASK_LIST <<< "${TASKS//,/ }"
cd "$ROOT"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$SIM_GPU \
  "$SIM_PY" sim/official_eval.py \
    --tasks "${TASK_LIST[@]}" --num-trials "$NUM_TRIALS" --split "$SPLIT" --seed "$SEED" \
    --log-dir "$LOG_DIR" \
    --s1-port "$S1_PORT" --s2-port "$S2_PORT" --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" \
    --norm-stats "$S1_CKPT/assets/robocasa_system1/norm_stats.json" \
    "${EXTRA[@]}" 2>&1 | tee "$LOG_DIR/logs/eval.log"
"$SIM_PY" sim/summarize.py --log-dir "$LOG_DIR" --split "$SPLIT" | tee "$LOG_DIR/summary.txt"
