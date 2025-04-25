#!/bin/bash

# # where your conda is installed
# CONDA_PATH="${HOME}/software/opt/miniconda3"
# # your conda environment name
# CONDA_ENV_NAME="torch"

# source ${CONDA_PATH}/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV_NAME}

# slurm partition
SLURM_PARTITION="tao"
# slurm node in the partition
SLURM_NODE="t000"

# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)

# you can specify a task name
# '-reset' means if the file in "output_path" (specified in config.yaml) already exits, it will be overwrote. you can delete it if you wish.
python eval/run.py  -task seg_fiber \
                    -gpu 0 \
                    -cfg eval/config/config.yaml \
                    -slurm \
                    -slurm_nodelist ${SLURM_NODE} \
                    -slurm_partition ${SLURM_PARTITION} \
                    -reset
