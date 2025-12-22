#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="python"
SLURM_PARTITION="compute"
SLURM_NODE="c003"
SLURM_NGPUS="4"
SEG_TASK="seg_fiber"
GPU_ID="0"

INPUT_PATH="/share/data/VISoR_Reconstruction/SIAT_SIAT/XuFang/Mouse_Brain/20250418_YY_BCP_CAMKII_T154_1/T154_1um.ims"
BG_THRES="150"
LEVEL="0"
CHANNEL="0"
PATCH_SIZE="300"
SLICE_THICKNESS="300"
KEEP_BRANCH="false"
CKPT_PATH="/share/home/liuy/project/SegFiber_dev/out/weights/for_C534/best_val_model_tiny.pth"

SAVE_FOLDER="/share/home/liuy/project/data/neurofly_data/db/SEED_T154_sliceSeg"
SAVE_PREFIX="SEED_T154_"

KEEP_BRANCH_FLAG=()
case "$(printf '%s' "${KEEP_BRANCH}" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes) KEEP_BRANCH_FLAG=(-keep_branch True) ;;
esac

SITE_PATCH_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${SITE_PATCH_DIR}"
}
trap cleanup EXIT

cat > "${SITE_PATCH_DIR}/sitecustomize.py" <<'PY'
import argparse
_orig_parse_args = argparse.ArgumentParser.parse_args
def _patched_parse_args(self, *args, **kwargs):
    ns = _orig_parse_args(self, *args, **kwargs)
    if hasattr(ns, "slice_thnickness") and not hasattr(ns, "slice_thickness"):
        setattr(ns, "slice_thickness", getattr(ns, "slice_thnickness"))
    return ns
argparse.ArgumentParser.parse_args = _patched_parse_args
PY

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SITE_PATCH_DIR}:${PROJECT_ROOT}"
else
  export PYTHONPATH="${SITE_PATCH_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"
fi

TOTAL_Z=13200
Z_CHUNK=300
NUM_SUBVOLUMES=$((TOTAL_Z / Z_CHUNK))
X_SIZE=12000
Y_SIZE=8000

if (( TOTAL_Z % Z_CHUNK != 0 )); then
  echo "Total z (${TOTAL_Z}) must be divisible by chunk thickness (${Z_CHUNK})." >&2
  exit 1
fi

mkdir -p "${SAVE_FOLDER}"
echo "Using output directory: ${SAVE_FOLDER}"
echo "Using filename prefix: ${SAVE_PREFIX}"
echo "Submitting ${NUM_SUBVOLUMES} jobs (ROI ${X_SIZE}x${Y_SIZE}x${Z_CHUNK})."

for slice_idx in $(seq -w 1 "${NUM_SUBVOLUMES}"); do
  numeric_idx=$((10#$slice_idx))
  z_start=$(( (numeric_idx - 1) * Z_CHUNK ))
  output_db="${SAVE_FOLDER}/${SAVE_PREFIX}z${slice_idx}.db"
  echo "Launching chunk ${slice_idx} (z ${z_start} -> $((z_start + Z_CHUNK))) -> ${output_db}"
  cmd=(
    "${PYTHON_BIN}" -m eval.eval
    -task "${SEG_TASK}_${slice_idx}"
    -gpu "${GPU_ID}"
    -slurm
    -slurm_nodelist "${SLURM_NODE}"
    -slurm_partition "${SLURM_PARTITION}"
    -slurm_ngpus "${SLURM_NGPUS}"
    -input_path "${INPUT_PATH}"
    -output_path "${output_db}"
    -bg_thres "${BG_THRES}"
    -level "${LEVEL}"
    -channel "${CHANNEL}"
    -patch_size "${PATCH_SIZE}"
    -slice_thnickness "${SLICE_THICKNESS}"
    -ckpt_path "${CKPT_PATH}"
    -roi 0 0 "${z_start}" "${X_SIZE}" "${Y_SIZE}" "${Z_CHUNK}"
  )
  if [[ ${#KEEP_BRANCH_FLAG[@]} -gt 0 ]]; then
    cmd+=("${KEEP_BRANCH_FLAG[@]}")
  fi
  "${cmd[@]}"
done
