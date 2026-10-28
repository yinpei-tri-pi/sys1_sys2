#!/usr/bin/env bash
# Download the evaluated System1 and System2 checkpoints from the Hugging Face Hub into checkpoints/.
#
#   bash scripts/download_checkpoints.sh            # HF_TOKEN=<token> if the repos are private
#
# Layout (the run scripts use these paths by default; the dir names carry the run name):
#   checkpoints/system1/f0717-270k-bs512-progact__progact_granfine_verbsimp_noexec/269999      ~12 GB
#   checkpoints/system2/system2-full-0804-qwen3vl-4b-...-ep3/checkpoint-17124                   ~9 GB
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/checkpoints.env"
command -v uvx >/dev/null || { echo "FATAL: uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
HF=(uvx --from "huggingface_hub>=1.0" hf)

for spec in "$S1_REPO $ROOT/$S1_CKPT_DEFAULT" "$S2_REPO $ROOT/$S2_CKPT_DEFAULT"; do
  read -r repo dir <<< "$spec"
  echo "[download] $repo -> $dir"
  mkdir -p "$dir"
  "${HF[@]}" download "$repo" --repo-type model --local-dir "$dir"
done
[[ -d "$ROOT/$S1_CKPT_DEFAULT/params" ]] || { echo "FATAL: System1 checkpoint incomplete (no params/)" >&2; exit 1; }
[[ -s "$ROOT/$S2_CKPT_DEFAULT/model.safetensors.index.json" ]] || { echo "FATAL: System2 checkpoint incomplete" >&2; exit 1; }
echo "[download] done"
