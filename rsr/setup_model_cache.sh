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

# Molmo is complete in the legacy shared cache. GroundingDINO and BERT are
# complete in the newer shared cache. Expose both through one rsr-local cache
# view without copying any model weights or modifying the public directories.
link_directory \
    /home/data/models/huggingface/hub/models--allenai--Molmo-7B-D-0924 \
    "$HUB_ROOT/models--allenai--Molmo-7B-D-0924"
link_directory \
    /home/data/datasets/.cache/hf/models--ShilongLiu--GroundingDINO \
    "$HUB_ROOT/models--ShilongLiu--GroundingDINO"
link_directory \
    /home/data/datasets/.cache/hf/models--bert-base-uncased \
    "$HUB_ROOT/models--bert-base-uncased"
link_directory \
    /home/data/models/huggingface/modules \
    "$CACHE_ROOT/modules"

mkdir -p "$HUB_ROOT/.locks"
