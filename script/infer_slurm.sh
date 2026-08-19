#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKSPACE=$(realpath "$SCRIPT_DIR/..")
CONFIG_PATH="$WORKSPACE/src/seg_fiber/model/config/template.yaml"
INPUT_PATH=""
OUTPUT_PATH="$WORKSPACE/out/segfiber.db"
CHECKPOINT=universal_tiny.pth
JOB_NAME=SegFiberInfer
SLURM_PARTITION=""
SLURM_NODE=""
NUM_NODES=1
NUM_CPUS_PER_TASK=2
NUM_GPUS_PER_NODE=4
LOAD_ENV=""
RESET=false

cd "$WORKSPACE"
INFER_COMMAND="segfiber infer --config \"$CONFIG_PATH\" --runtime ddp --input \"$INPUT_PATH\" --output \"$OUTPUT_PATH\" --checkpoint \"$CHECKPOINT\""
if [ "$RESET" = true ]; then INFER_COMMAND="$INFER_COMMAND --reset"; fi
COMMAND=(
    python -m seg_fiber.model.runtime.slurm_submit
    --job-name "$JOB_NAME"
    --command "$INFER_COMMAND"
    --config "$CONFIG_PATH"
    --num-nodes "$NUM_NODES"
    --num-cpus "$NUM_CPUS_PER_TASK"
    --num-gpus "$NUM_GPUS_PER_NODE"
    --load-env "$LOAD_ENV"
)
if [ -n "$SLURM_PARTITION" ]; then COMMAND+=(--partition "$SLURM_PARTITION"); fi
if [ -n "$SLURM_NODE" ]; then COMMAND+=(--node "$SLURM_NODE"); fi
"${COMMAND[@]}"
