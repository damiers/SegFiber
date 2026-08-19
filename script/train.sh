#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKSPACE=$(realpath "$SCRIPT_DIR/..")
CONFIG_PATH="$WORKSPACE/src/seg_fiber/model/config/template.yaml"

cd "$WORKSPACE"
segfiber train --config "$CONFIG_PATH"
