# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import argparse
from eval.utils.distributed import init_dist_node, init_dist_gpu
from eval.dataset.parallelDataset import parallelDataset
from eval.utils.sqliteDBIO import sqliteDBIO
from eval.seger import Seger

import submitit, random, sys
import os

from pprint import pprint

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === arg praser ===
def parse_args():
    parser = argparse.ArgumentParser(description='Seger')

    # === GENERAL === #
    parser.add_argument('-task', type=str, default="SegFiber",
                                            help='task name')
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type =str,
                                            help='Configuration file')

    # === PATHS === #
    parser.add_argument('-input_path', type=str, default="data",
                                            help='input path')
    parser.add_argument('-output_path', type=str, default="data",
                                            help='output path')
    
    parser.add_argument('-bg_thres', type=int, default=300,
                                            help='background intensity value')
    
    parser.add_argument('-channel', type=int, default=0,
                                            help='image channel')
    
    parser.add_argument('-patch_size', type=int, default=0,
                                            help='the size of the cube accepted by the segmentation model')
                                            
    parser.add_argument('-slice_thnickness', type=int, default=300,
                                            help='thickness of a brain slice')
    
    parser.add_argument('-roi', type=int, nargs='+', default=None,
                                            help='roi')

    # === SLURM === #
    parser.add_argument('-slurm', action='store_true', default=False,
                                            help='Submit with slurm')
    parser.add_argument('-slurm_ngpus', type=int, default = 2,
                                            help='num of gpus per node')
    parser.add_argument('-slurm_nnodes', type=int, default = 1,
                                            help='number of nodes')
    parser.add_argument('-slurm_nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-slurm_partition', type=str, default = "general",
                                            help='slurm partition')
    parser.add_argument('-slurm_timeout', type=int, default = 2800,
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')


    args = parser.parse_args()

    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            import ruamel.yaml as yaml
            from ruamel.yaml import YAML
            yaml = YAML(typ='safe', pure=True)
            yml = yaml.load(f)
        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    # path validation
    if os.path.splitext(args.output_path)[-1]:
        file = os.path.basename(args.output_path)
        if not str.endswith(file, 'db'):
            file = f'{str.split(file, ".")[0]}.db'
        directory = os.path.dirname(args.output_path)
    else:
        file = f'segerOut_{str.split(os.path.basename(args.input_path), ".")[0]}.db'
        directory = args.output_path

    check_dir(directory)
    args.output_path = os.path.join(directory, file)    
    
    if os.path.exists(args.output_path) and args.reset:
        os.remove(args.output_path)

    return args

def main():
    args = parse_args()
    args.port = random.randint(49152,65535)
    
    pprint(vars(args))

    if args.slurm:
        args.slurm_log_dir = os.path.join(os.path.dirname(args.output_path), 'slurm_log/%j')
        executor = submitit.AutoExecutor(folder=args.slurm_log_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=12*args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.task)
        trainer = WORKER_SLURM(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")


    else:
        init_dist_node(args)
        mp.spawn(WORKER, args = (args,), nprocs = args.ngpus_per_node)
	

# === worker ===
def WORKER(gpu, args):
    pprint(vars(args))

    # === SET ENV === #
    init_dist_gpu(gpu, args)

    # === DATA === #
    dataset = parallelDataset(
        args.input_path, 
        patch_size=args.patch_size, 
        slice_thickness=args.slice_thickness, 
        level=args.level, 
        channel=args.channel, 
        roi=args.roi
    )

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=args.world_size, rank=args.rank, seed=31, drop_last=False)
    def custom_collate(batch):
        img_patch = batch[0][0]
        offset = batch[0][1]
        re_batch = batch[0][2]
        return img_patch, offset, re_batch
    dataloader = DataLoader(
        dataset=dataset, 
        sampler=sampler,
        batch_size=1, 
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate
    )
    seger = Seger(ckpt_path=None, bg_thres=args.bg_thres, cuda_device_id=args.gpu)
    dbio = sqliteDBIO(args.output_path)
    _, seg_version = dbio.get_max_sid_version()

    with torch.no_grad():
        for img_patch, offset, re_batch in dataloader:
            # img_patch = img_patch.cuda(args.gpu)
            # offset = offset.cpu().numpy().astype(int).tolist()[0]
            segs = seger.process(img_patch, offset, re_batch)

            gathered_results = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered_results, segs)
            if args.rank == 0:
                for segs in gathered_results:
                    if segs is not None:
                            dbio.segs2db(segs, version=seg_version)
    dist.destroy_process_group()

class WORKER_SLURM(object):
    def __init__(self, args):
        self.args = args
    def __call__(self):
        init_dist_node(self.args)
        WORKER(None, self.args)

if __name__ == "__main__":
    main()
