#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKSPACE=$(realpath "$SCRIPT_DIR/..")
CONFIG_PATH="$WORKSPACE/src/seg_fiber/model/config/template.yaml"
INPUT_PATH=""
OUTPUT_PATH="$WORKSPACE/out/segfiber.db"
CHECKPOINT=universal_tiny.pth

cd "$WORKSPACE"
segfiber infer \
    --config "$CONFIG_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --checkpoint "$CHECKPOINT"
