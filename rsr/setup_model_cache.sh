#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${1:-$ROOT_DIR/rsr/data/model_cache}"
HUB_ROOT="$CACHE_ROOT/hub"

mkdir -p "$HUB_ROOT" "$CACHE_ROOT/assets" "$CACHE_ROOT/xet"

link_directory() {
    local source="$1"
    local destination="$2"
    if [[ ! -e "$source" ]]; then
        echo "Required public model cache is missing: $source" >&2
        exit 1
    fi
    if [[ -e "$destination" && ! -L "$destination" ]]; then
        echo "Refusing to replace non-symlink cache entry: $destination" >&2
        exit 1
    fi
    ln -sfn "$source" "$destination"
}

find_directory() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -e "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Required public model cache is missing: $label" >&2
    return 1
}

# Discover complete public caches on either lab account, then expose them
# through one rsr-local symlink view. No model files are copied or modified.
MOLMO_CACHE="$(find_directory Molmo \
    "${RSR_MOLMO_CACHE_DIR:-}" \
    "$HOME/.cache/huggingface/hub/models--allenai--Molmo-7B-D-0924" \
    /home/data/models/huggingface/hub/models--allenai--Molmo-7B-D-0924 \
    /home/data/datasets/.cache/hf/models--allenai--Molmo-7B-D-0924)"
DINO_CACHE="$(find_directory GroundingDINO \
    "${RSR_DINO_CACHE_DIR:-}" \
    "$HOME/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO" \
    /home/data/datasets/.cache/hf/models--ShilongLiu--GroundingDINO \
    /home/data/models/huggingface/hub/models--ShilongLiu--GroundingDINO)"
BERT_CACHE="$(find_directory bert-base-uncased \
    "${RSR_BERT_CACHE_DIR:-}" \
    "$HOME/.cache/huggingface/hub/models--bert-base-uncased" \
    /home/data/datasets/.cache/hf/models--bert-base-uncased \
    /home/data/models/huggingface/hub/models--bert-base-uncased)"
MODULES_CACHE="$(find_directory HuggingFace-modules \
    "${RSR_HF_MODULES_DIR:-}" \
    "$HOME/.cache/huggingface/modules" \
    /home/data/models/huggingface/modules \
    /home/data/datasets/.cache/hf/modules)"

link_directory "$MOLMO_CACHE" "$HUB_ROOT/models--allenai--Molmo-7B-D-0924"
link_directory "$DINO_CACHE" "$HUB_ROOT/models--ShilongLiu--GroundingDINO"
link_directory "$BERT_CACHE" "$HUB_ROOT/models--bert-base-uncased"
link_directory "$MODULES_CACHE" "$CACHE_ROOT/modules"

mkdir -p "$HUB_ROOT/.locks"
