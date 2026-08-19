#!/bin/bash
${PARTITION_OPTION}
${NODE_OPTION}

#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=${NUM_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${NUM_CPUS_PER_TASK}
#SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_PATH}_out.log
#SBATCH --error=${LOG_PATH}_err.log

set -euo pipefail

${LOAD_ENV}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

srun ${COMMAND}
