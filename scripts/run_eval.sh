#!/usr/bin/env bash
# One evaluation stack: System1 server + System2 server + the RoboCasa episode loop.
#
#   bash scripts/run_eval.sh --s1-ckpt <s1 step dir> --s2-ckpt <s2 checkpoint dir> \
#       --s1-gpu 0 --s2-gpu 1 --tasks atomic_seen --num-trials 50 --log-dir results/<run>
#
# --protocol    official (default): leaderboard protocol, sim/official_eval.py (target scenes, seed,
#               --num-trials fresh resets per task) | target: the 30 recorded target-split episodes
#               per task, sim/target_eval.py (shared work queue; --dataset-root = LeRobot target root)
# --s1-ckpt / --s2-ckpt  default: the downloaded checkpoints (scripts/checkpoints.env)
# --log-dir     default: results/<protocol>__<run name>, e.g. results/official__s1-progact270k_s2-qwen3vl-4b-full-ep3-17124
# --tasks       comma-separated task names and/or groups: atomic_seen,composite_seen,composite_unseen,all
# --sim-gpu     GPU for MuJoCo EGL rendering (default: the System1 GPU)
# --skip-servers  reuse servers already listening on --s1-port/--s2-port
# --name        log subdir, <log-dir>/logs/<name>/ (several stacks sharing one --log-dir)
# Anything after `--` is passed to the protocol driver (e.g. -- --no-task-lang-steer).
# Servers are stopped when the evaluation exits. Logs: <log-dir>/logs/[<name>/].
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

S1_CKPT="" S2_CKPT="" S1_GPU=0 S2_GPU=1 SIM_GPU="" S1_PORT=8060 S2_PORT=8100
S1_MEM=0.35 S2_MEM=0.45 TASKS="atomic_seen" NUM_TRIALS=50 SPLIT=target SEED=7 LOG_DIR=""
SKIP_SERVERS=0 GPU_MODE=${DOCKER_GPU_MODE:-gpus} NAME="" PROTOCOL=official
DATASET_ROOT=${ROBOCASA_LEROBOT_ROOT:-$HOME/robocasa_dataset}/v1.0/target EXTRA=()
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
    --name) NAME=$2; shift 2 ;;
    --protocol) PROTOCOL=$2; shift 2 ;;
    --dataset-root) DATASET_ROOT=$2; shift 2 ;;
    --docker-gpu-mode) GPU_MODE=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n 2,18p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$PROTOCOL" == official || "$PROTOCOL" == target ]] || { echo "FATAL: --protocol official|target" >&2; exit 2; }
source "$ROOT/scripts/checkpoints.env"
S1_CKPT=${S1_CKPT:-$ROOT/$S1_CKPT_DEFAULT}; S2_CKPT=${S2_CKPT:-$ROOT/$S2_CKPT_DEFAULT}
[[ -d "$S1_CKPT" && -d "$S2_CKPT" ]] \
  || { echo "FATAL: checkpoints not found ($S1_CKPT, $S2_CKPT); run scripts/download_checkpoints.sh" >&2; exit 2; }
S1_CKPT=$(cd "$S1_CKPT" && pwd); S2_CKPT=$(cd "$S2_CKPT" && pwd)
SIM_GPU=${SIM_GPU:-$S1_GPU}
SIM_PY="$ROOT/sim/.venv/bin/python"
[[ -x "$SIM_PY" ]] || { echo "FATAL: $SIM_PY missing; run scripts/setup.sh" >&2; exit 1; }
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR=$ROOT/results/$("$SIM_PY" "$ROOT/sim/run_name.py" --protocol "$PROTOCOL" --s1-dir "$S1_CKPT" \
    --s2-dir "$S2_CKPT" --num-trials "$NUM_TRIALS" --split "$SPLIT" --seed "$SEED" 2>/dev/null | tail -n 1)
fi
mkdir -p "$LOG_DIR"; LOG_DIR=$(cd "$LOG_DIR" && pwd)
LOGS=$LOG_DIR/logs${NAME:+/$NAME}; mkdir -p "$LOGS"
echo "[run_eval] results -> $LOG_DIR"

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
    --mem-frac "$S1_MEM" > "$LOGS/system1.log" 2>&1 &
  pids+=("$!")
  # System1 first, System2 once it answers: JAX preallocates its whole --mem-frac share while loading,
  # and vLLM sizes its KV cache from the memory free during its own profiling. If System1 allocates
  # inside that window (same GPU), vLLM counts it as its own overhead and fails with "No available
  # memory for the cache blocks".
  echo "[run_eval] starting System1 :$S1_PORT; System2 follows once it is up (logs in $LOGS)"
  for _ in $(seq 1 60); do
    [[ $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$S1_PORT/healthz" || true) == 200 ]] && break
    kill -0 "${pids[0]}" 2>/dev/null || { echo "FATAL: System1 exited; see $LOGS/system1.log" >&2; exit 1; }
    sleep 10
  done
  bash "$ROOT/scripts/serve_system2.sh" --ckpt "$S2_CKPT" --gpu "$S2_GPU" --port "$S2_PORT" \
    --mem-frac "$S2_MEM" --docker-gpu-mode "$GPU_MODE" > "$LOGS/system2.log" 2>&1 &
  pids+=("$!")
  # A cold vLLM start (image import + torch.compile + CUDA graphs) can take ~20 min.
  echo "[run_eval] waiting for System1 :$S1_PORT and System2 :$S2_PORT (logs in $LOGS)"
  for _ in $(seq 1 240); do
    # /healthz is the System1 server's HTTP health route; a bare TCP probe would make it log a
    # failed websocket handshake (three tracebacks) on every poll.
    s1=$( [[ $(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$S1_PORT/healthz" || true) == 200 ]] && echo up || echo down)
    s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$S2_PORT/v1/models" || true)
    [[ "$s1" == up && "$s2" == 200 ]] && break
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "FATAL: a server exited; see $LOGS" >&2; exit 1; }; done
    sleep 10
  done
  [[ "$s1" == up && "$s2" == 200 ]] || { echo "FATAL: servers not ready after 40 min" >&2; exit 1; }
fi
read -ra TASK_LIST <<< "${TASKS//,/ }"
cd "$ROOT"
if [[ "$PROTOCOL" == official ]]; then
  echo "[run_eval] servers ready; official protocol tasks=[$TASKS] trials=$NUM_TRIALS split=$SPLIT seed=$SEED"
  DRIVER=(sim/official_eval.py --num-trials "$NUM_TRIALS" --split "$SPLIT" --seed "$SEED")
  SUMMARY=(--protocol official --split "$SPLIT")
else
  echo "[run_eval] servers ready; target-split protocol tasks=[$TASKS] dataset=$DATASET_ROOT"
  DRIVER=(sim/target_eval.py --dataset-root "$DATASET_ROOT" --worker-name "${NAME:-stack}-$(hostname)")
  SUMMARY=(--protocol target)
fi
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$SIM_GPU \
  "$SIM_PY" "${DRIVER[@]}" --tasks "${TASK_LIST[@]}" --log-dir "$LOG_DIR" \
    --s1-port "$S1_PORT" --s2-port "$S2_PORT" --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" \
    --norm-stats "$S1_CKPT/assets/robocasa_system1/norm_stats.json" \
    "${EXTRA[@]}" 2>&1 | tee "$LOGS/eval.log"
"$SIM_PY" sim/summarize.py --log-dir "$LOG_DIR" "${SUMMARY[@]}" | tee "$LOGS/summary.txt"
