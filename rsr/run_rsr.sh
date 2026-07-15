#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# API evaluation is intentionally direct. Model loading uses the public cache
# in offline mode, so inherited proxy variables are not needed.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

PYTHON="${PYTHON:-/home/qiuguanhe/miniconda3/envs/freegrasp/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && " $* " != *" --localization-only "* && " $* " != *" --molmo-only "* ]]; then
    echo "OPENAI_API_KEY is not set. Export it in the current shell before reasoning." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RSR_HF_HOME="${RSR_HF_HOME:-$ROOT_DIR/rsr/data/model_cache}"
bash "$ROOT_DIR/rsr/setup_model_cache.sh" "$RSR_HF_HOME"

# Do not inherit stale Hugging Face variables from the interactive shell. The
# complete public models live in two different caches and are combined by the
# rsr-local symlink view above.
export HF_HOME="$RSR_HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
# Remove deprecated overrides so Transformers derives its cache from the fixed
# HF_HOME/HF_HUB_CACHE above without emitting legacy-cache warnings.
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="${TORCH_HOME:-/home/data/models/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$PYTHON" -u -m rsr.run "$@"
