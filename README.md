# SegFiber

1. install conda & create a conda environment

   ```bash
   conda create -n ${CONDA_ENV} python=3.10
   ```
2. clone this repo in local & install package

   ```bash
   git clone https://github.com/damiers/SegFiber.git SegFiber

   cd SegFiber

   conda activate ${CONDA_ENV}

   pip install -e .

   # or with pytorch
   pip install -e .[pytorch]
   ```
3. edit algorithm releated parameters in the `config.yaml`, for example:

   ```yaml
   # background threshold
   bg_thres: 300
   # resolution level
   level: 2
   # channel number, 0 means 488
   channel: 0
   # cube size accepted by the segmentation model
   chunk_size: 300
   # thickness of one brain slice
   splice: 300

   # ROI you wanna segment, 'null' means the entire volume to be processed
   # format: [x_start, y_start, z_start, x_size, y_size, z_size]
   # roi: [0,0,0,128,128,128]
   roi: null

   # where your image data
   input_path: 'test/data/test.tif'
   # where the database file is output
   output_path: 'test/out/test.db'
   ```
4. edit SLURM system releated parameters in the `run.sh`, for example:

   ```bash
   #!/bin/bash

   # where your conda is installed
   CONDA_PATH='~/software/opt/miniconda3'
   # your conda environment name
   CONDA_ENV_NAME="torch"

   source ${CONDA_PATH}/etc/profile.d/conda.sh
   conda activate ${CONDA_ENV_NAME}

   # slurm partition
   SLURM_PARTITION="compute"
   # slurm node in the partition
   SLURM_NODE="c001"

   # you can specify a task name
   # '-reset' means if the file in "output_path" (specified in config.yaml) already exits, it will be overwrote. you can delete it if you wish.
   python run.py -task seg_fiber \
                 -gpu 0 \
                 -cfg config.yaml \
                 -slurm \
                 -slurm_nodelist ${SLURM_NODE} \
                 -slurm_partition ${SLURM_PARTITION} \
                 -reset
   ```
