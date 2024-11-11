python run.py -task seg_fiber \
              -gpu 0 \
              -out out \
              -cfg config.yaml \
              -slurm \
              -slurm_nodelist c001 \
              -slurm_partition compute \
              -reset