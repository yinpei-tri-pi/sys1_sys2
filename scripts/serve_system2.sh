#!/usr/bin/env bash
# Serve ONE System2 (Qwen-VL planner) checkpoint on ONE GPU with vLLM in Docker. Blocks.
#
#   bash scripts/serve_system2.sh --ckpt <run>/checkpoint-17124 --gpu 1 --port 8100 [--mem-frac 0.45]
#
# --ckpt   a FULL (merged) HF checkpoint dir; bare LoRA adapters are rejected.
# --image  defaults to the image built from system2/Dockerfile (see README).
# --docker-gpu-mode  gpus (default: `--gpus device=N`) | runtime (`--runtime=nvidia` +
#          NVIDIA_VISIBLE_DEVICES=N, for CDI/snap Docker that rejects --gpus).
# Probe:   curl -s http://127.0.0.1:8100/v1/models
#
# Every serve flag and media env var is copied from sys2_train_eval/scripts/serve_system2_vllm.sh,
# which matches the validated System2 dataset eval, so rollouts decode video identically:
#   * --media-io-kwargs / --limit-mm-per-prompt / --allowed-local-media-path must be present or
#     video requests fail or silently re-sample.
#   * FPS / FPS_MAX_FRAMES / FRAME_FACTOR / *_MAX_TOKEN_NUM pin qwen-vl-utils' globals.
#   * The client still sends per-request frame counts (sim/loop/sys2_client.py).
set -euo pipefail

CKPT="" GPU=0 PORT=8100 MEM_FRAC=0.45
IMAGE=${SYS2_IMAGE:-sys1-sys2/system2-vllm:0.19.1}
GPU_MODE=${DOCKER_GPU_MODE:-gpus}
SERVED_NAME=system2-full MAX_MODEL_LEN=8192 MAX_NUM_SEQS=4
MEDIA_DIR=${MEDIA_DIR:-/tmp/sys2_media}   # client clip/scene media (file:// mode); harmless otherwise
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --mem-frac) MEM_FRAC=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --served-name) SERVED_NAME=$2; shift 2 ;;
    --docker-gpu-mode) GPU_MODE=$2; shift 2 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CKPT" ]] || { echo "FATAL: --ckpt is required" >&2; exit 2; }
CKPT=$(cd "$CKPT" && pwd)
if [[ -s "$CKPT/adapter_model.safetensors" && ! -s "$CKPT/model.safetensors" \
      && ! -s "$CKPT/model.safetensors.index.json" ]]; then
  echo "FATAL: $CKPT is an unmerged LoRA adapter; merge it into a full checkpoint first." >&2
  exit 1
fi
case "$GPU_MODE" in
  gpus) GPU_ARGS=(--gpus "device=$GPU") ;;
  runtime) GPU_ARGS=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=$GPU") ;;
  *) echo "FATAL: --docker-gpu-mode must be gpus|runtime" >&2; exit 2 ;;
esac
mkdir -p "$MEDIA_DIR"

echo "[system2] ckpt=$CKPT gpu=$GPU port=$PORT mem_frac=$MEM_FRAC image=$IMAGE"
exec docker run --rm --name "sys1sys2-system2-$PORT" \
  "${GPU_ARGS[@]}" --ipc=host --shm-size=32g --network host \
  --entrypoint /bin/bash \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_VIDEO_LOADER_BACKEND=opencv \
  -e FORCE_QWENVL_VIDEO_READER=torchvision \
  -e FPS=4 -e FPS_MIN_FRAMES=2 -e FPS_MAX_FRAMES=32 -e FRAME_FACTOR=2 \
  -e SPATIAL_MERGE_SIZE=2 -e IMAGE_MAX_TOKEN_NUM=192 -e VIDEO_MAX_TOKEN_NUM=192 \
  -e HOME=/tmp/sys2_runtime/home -e HF_HOME=/tmp/sys2_runtime/hf \
  -e XDG_CACHE_HOME=/tmp/sys2_runtime/cache -e TRITON_CACHE_DIR=/tmp/sys2_runtime/triton \
  -v "$CKPT:/opt/ml/input/data/model:ro" \
  -v "$MEDIA_DIR:$MEDIA_DIR:rw" \
  "$IMAGE" -lc "
    set -euo pipefail
    mkdir -p /tmp/sys2_runtime/{home,hf,cache,triton}
    exec /usr/local/bin/vllm serve /opt/ml/input/data/model \
      --host 0.0.0.0 --port $PORT \
      --served-model-name $SERVED_NAME \
      --dtype bfloat16 \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $MEM_FRAC \
      --max-num-seqs $MAX_NUM_SEQS \
      --max-num-batched-tokens 16384 \
      --mm-processor-cache-gb 1 \
      --media-io-kwargs '{\"video\":{\"num_frames\":-1,\"frame_recovery\":true}}' \
      --limit-mm-per-prompt '{\"image\":3,\"video\":1}' \
      --allowed-local-media-path $MEDIA_DIR
  "
