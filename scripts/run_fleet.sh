#!/usr/bin/env bash
# Full benchmark on several GPUs: one complete stack (System1 + System2 + simulator) per GPU. The
# stacks pull tasks from a shared queue (see --schedule), all write into ONE --log-dir, and the
# results are summarized at the end.
#
#   bash scripts/run_fleet.sh                      # the full benchmark: 50 tasks x 50 trials, every GPU
#   bash scripts/run_fleet.sh --num-trials 2       # 100-episode pilot
#   bash scripts/run_fleet.sh --protocol target    # 30 recorded target-split episodes per task
#
# Defaults: --gpus all GPUs, --tasks all, --num-trials 50, the downloaded checkpoints
# (scripts/checkpoints.env), --log-dir results/<protocol>__<run name>.
#
# --protocol official (default) | target (30 recorded target-split episodes per task)
# --schedule queue (default): stacks pull work from one shared queue in <log-dir>/claims/ -- whole
#            tasks, longest horizon first, for official; single episodes for target. More stacks can
#            join the same --log-dir later (other --gpus, distinct --s1-base-port/--s2-base-port and
#            --stack-prefix). shard: official only, stack i runs tasks i, i+N, ... (fixed split)
#
# --stacks-per-gpu K  K stacks per listed GPU (2 on 80 GB cards: --s1-mem-frac 0.13 --s2-mem-frac 0.20);
#            a GPU's later stacks start once its first stack's servers are up
# --gpus     one stack per listed GPU; System1, System2 and the simulator share it (80 GB cards:
#            default --s1-mem-frac 0.32 --s2-mem-frac 0.42). For smaller cards put System2 on its
#            own GPUs with --s2-gpus (same length as --gpus, matched element-wise).
# Stack i listens on ports 8060+i / 8100+i and logs to <log-dir>/logs/stack<i>/. Anything after
# `--` goes to sim/official_eval.py. A task whose stats.json exists is skipped, so an interrupted
# fleet is relaunched with the same command (an interrupted task restarts from trial 0).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

S1_CKPT="" S2_CKPT="" GPUS="" S2_GPUS="" TASKS=all NUM_TRIALS=50 LOG_DIR=""
S1_MEM=0.32 S2_MEM=0.42 S1_BASE=8060 S2_BASE=8100 PROTOCOL=official SCHEDULE=queue PREFIX=stack PER_GPU=1 PASS=() EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --s1-ckpt) S1_CKPT=$2; shift 2 ;;
    --s2-ckpt) S2_CKPT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --s2-gpus) S2_GPUS=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --num-trials) NUM_TRIALS=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --s1-mem-frac) S1_MEM=$2; shift 2 ;;
    --s2-mem-frac) S2_MEM=$2; shift 2 ;;
    --s1-base-port) S1_BASE=$2; shift 2 ;;
    --s2-base-port) S2_BASE=$2; shift 2 ;;
    --protocol) PROTOCOL=$2; PASS+=("$1" "$2"); shift 2 ;;
    --stack-prefix) PREFIX=$2; shift 2 ;;
    --stacks-per-gpu) PER_GPU=$2; shift 2 ;;
    --schedule) SCHEDULE=$2; shift 2 ;;
    --split|--seed|--docker-gpu-mode|--dataset-root) PASS+=("$1" "$2"); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n 2,22p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
GPUS=${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}   # default: every GPU
source "$ROOT/scripts/checkpoints.env"
S1_CKPT=${S1_CKPT:-$ROOT/$S1_CKPT_DEFAULT}; S2_CKPT=${S2_CKPT:-$ROOT/$S2_CKPT_DEFAULT}
[[ -d "$S1_CKPT" && -d "$S2_CKPT" ]] \
  || { echo "FATAL: checkpoints not found ($S1_CKPT, $S2_CKPT); run scripts/download_checkpoints.sh" >&2; exit 2; }
S1_CKPT=$(cd "$S1_CKPT" && pwd); S2_CKPT=$(cd "$S2_CKPT" && pwd)
if [[ -z "$LOG_DIR" ]]; then
  _split=target _seed=7
  for ((k = 0; k < ${#PASS[@]}; k++)); do
    [[ "${PASS[$k]}" == --split ]] && _split=${PASS[$((k + 1))]}
    [[ "${PASS[$k]}" == --seed ]] && _seed=${PASS[$((k + 1))]}
  done
  LOG_DIR=$ROOT/results/$("$ROOT/sim/.venv/bin/python" "$ROOT/sim/run_name.py" --protocol "$PROTOCOL" \
    --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" --num-trials "$NUM_TRIALS" --split "$_split" --seed "$_seed" 2>/dev/null | tail -n 1)
fi
IFS=',' read -ra G1 <<< "$GPUS"
IFS=',' read -ra G2 <<< "${S2_GPUS:-$GPUS}"
# --stacks-per-gpu K: K stacks on every listed GPU. The simulator is CPU-bound and the GPUs sit
# mostly idle, so two stacks per 80 GB card (with smaller memory fractions) nearly double throughput.
if (( PER_GPU > 1 )); then
  B1=("${G1[@]}"); B2=("${G2[@]}")
  for ((r = 1; r < PER_GPU; r++)); do G1+=("${B1[@]}"); G2+=("${B2[@]}"); done
fi
[[ ${#G1[@]} -eq ${#G2[@]} ]] || { echo "FATAL: --gpus and --s2-gpus must have the same length" >&2; exit 2; }
N=${#G1[@]}
mkdir -p "$LOG_DIR"; LOG_DIR=$(cd "$LOG_DIR" && pwd)

pids=()
cleanup() { trap - EXIT INT TERM; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup INT TERM

[[ "$SCHEDULE" == queue || "$SCHEDULE" == shard ]] || { echo "FATAL: --schedule queue|shard" >&2; exit 2; }
T0=$(date +%s); START_ISO=$(date -Is)
echo "[fleet] start $START_ISO on $(hostname)"
echo "[fleet] $N stacks, protocol=$PROTOCOL schedule=$SCHEDULE tasks=[$TASKS] trials=$NUM_TRIALS -> $LOG_DIR"
# A stack that shares a GPU with an earlier one starts only once that stack's servers answer: vLLM
# sizes its KV cache from the memory free while it profiles, so two concurrent starts on one card can
# miscount each other's allocation.
wait_ready() {   # wait_ready <stack index>
  local j=$1 s1 s2
  for _ in $(seq 1 240); do
    s1=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((S1_BASE + j))/healthz" || true)
    s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((S2_BASE + j))/v1/models" || true)
    [[ "$s1" == 200 && "$s2" == 200 ]] && return 0
    kill -0 "${pids[$j]}" 2>/dev/null || return 0   # that stack died; do not block the rest
    sleep 10
  done
}
for i in $(seq 0 $((N - 1))); do
  for ((j = 0; j < i; j++)); do
    if [[ "${G1[$j]}" == "${G1[$i]}" || "${G2[$j]}" == "${G2[$i]}" || "${G1[$j]}" == "${G2[$i]}" || "${G2[$j]}" == "${G1[$i]}" ]]; then
      echo "[fleet] $PREFIX$i shares a GPU with $PREFIX$j; waiting for its servers"; wait_ready "$j"
    fi
  done
  mkdir -p "$LOG_DIR/logs/$PREFIX$i"
  SHARD=()
  if [[ "$PROTOCOL" == official ]]; then
    [[ "$SCHEDULE" == shard ]] && SHARD=(--shard "$i/$N") || SHARD=(--task-queue --worker-name "$PREFIX$i-$(hostname)")
  fi
  bash "$ROOT/scripts/run_eval.sh" \
    --s1-ckpt "$S1_CKPT" --s2-ckpt "$S2_CKPT" --s1-gpu "${G1[$i]}" --s2-gpu "${G2[$i]}" \
    --s1-port $((S1_BASE + i)) --s2-port $((S2_BASE + i)) --s1-mem-frac "$S1_MEM" --s2-mem-frac "$S2_MEM" \
    --tasks "$TASKS" --num-trials "$NUM_TRIALS" --log-dir "$LOG_DIR" --name "$PREFIX$i" "${PASS[@]}" \
    -- "${SHARD[@]}" "${EXTRA[@]}" > "$LOG_DIR/logs/$PREFIX$i/fleet.log" 2>&1 &
  pids+=("$!")
  echo "[fleet] $PREFIX$i: gpu ${G1[$i]} (S2 on ${G2[$i]})  S1 :$((S1_BASE + i))  S2 :$((S2_BASE + i))  log $LOG_DIR/logs/$PREFIX$i/"
done
echo "[fleet] monitor: tail -f $LOG_DIR/logs/$PREFIX*/eval.log   browse: python viewer/server.py --log-dir $LOG_DIR"
fail=0
for i in $(seq 0 $((N - 1))); do wait "${pids[$i]}" || { echo "[fleet] $PREFIX$i FAILED (see $LOG_DIR/logs/$PREFIX$i/)" >&2; fail=1; }; done
T1=$(date +%s); SECS=$((T1 - T0))
printf '{"host": "%s", "start": "%s", "end": "%s", "seconds": %d, "hours": %.2f, "stacks": %d, "protocol": "%s", "schedule": "%s", "tasks": "%s", "num_trials": %s, "failed_stacks": %d}\n' \
  "$(hostname)" "$START_ISO" "$(date -Is)" "$SECS" "$(echo "$SECS/3600" | bc -l)" "$N" "$PROTOCOL" "$SCHEDULE" "$TASKS" "$NUM_TRIALS" "$fail" \
  | tee "$LOG_DIR/timing.json"
echo "[fleet] end $(date -Is): wall time $((SECS / 3600))h $((SECS % 3600 / 60))m $((SECS % 60))s"
"$ROOT/sim/.venv/bin/python" "$ROOT/sim/summarize.py" --log-dir "$LOG_DIR" --protocol "$PROTOCOL" | tee "$LOG_DIR/summary.txt"
exit $fail
