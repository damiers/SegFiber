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

import os, sys
import argparse
from tqdm import tqdm
import submitit
import random
from pprint import pprint

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from eval.utils.distributed import init_dist_node, init_dist_gpu
from eval.dataset.parallelDataset import parallelDataset
from eval.utils.neurodb_sqlite import NeurodbSQLite
from eval.seger import Seger

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === arg praser ===
def parse_args(args_debug=None):
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
    
    parser.add_argument('-level', type=int, default=0,
                                            help='image resolution level')

    parser.add_argument('-channel', type=int, default=0,
                                            help='image channel')
    
    parser.add_argument('-patch_size', type=int, default=0,
                                            help='the size of the cube accepted by the segmentation model')
                                            
    parser.add_argument('-slice_thnickness', type=int, default=300,
                                            help='thickness of a brain slice')
    
    parser.add_argument('-roi', type=int, nargs='+', default=None,
                                            help='roi')
    
    parser.add_argument('-keep_branch', type=bool, default=False,
                                            help='keep branch in segmentation output')
    
    parser.add_argument('-ckpt_path', type=str, default=None,
                                            help='pretrained checkpoint path')

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

    if args_debug:
        args = parser.parse_args(args_debug)
    else:
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

# === worker ===
def __custom_collate__(batch):
    img_patch = batch[0][0]
    offset = batch[0][1]
    re_batch = batch[0][2]
    return img_patch, offset, re_batch

def WORKER(gpu, args):
    pprint(vars(args))

    # === SET ENV === #
    if hasattr(args, 'world_size') and args.world_size > 1:
        init_dist_gpu(gpu, args)
        use_distributed = True
    else:
        # Single process case
        use_distributed = False
        args.rank = 0
        args.world_size = 1
        if gpu is not None:
            args.gpu = gpu
        elif not hasattr(args, 'gpu'):
            args.gpu = 0


    # === DATA === #
    dataset = parallelDataset(
        args.input_path, 
        patch_size=args.patch_size, 
        slice_thickness=args.slice_thickness, 
        level=args.level, 
        channel=args.channel, 
        roi=args.roi
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=args.world_size, rank=args.rank, seed=31, drop_last=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset, 
        sampler=sampler,
        batch_size=1, 
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=__custom_collate__
    )
    ckpt_path = args.ckpt_path if hasattr(args, 'ckpt_path') else None
    seger = Seger(ckpt_path=ckpt_path, bg_thres=args.bg_thres, cuda_device_id=args.gpu)
    neurodb = NeurodbSQLite(args.output_path)
    _, seg_version = neurodb.get_max_sid_version()

    if args.rank == 0:
        pbar = tqdm(total=len(dataset), desc="GlobalProgressBar")  

    with torch.no_grad():
        for img_patch, offset, re_batch in dataloader:
            segs = seger.process(img_patch, offset, re_batch, keep_branch=args.keep_branch)
            if use_distributed:
                gathered_results = [None for _ in range(args.world_size)]
                dist.all_gather_object(gathered_results, segs)
                if args.rank == 0:
                    pbar.update(args.world_size)
                    for segs in gathered_results:
                        if segs is not None:
                                neurodb.segs2db(segs, version=seg_version)
            else:
                if args.rank == 0:
                    pbar.update(1)
                if segs is not None:
                    neurodb.segs2db(segs, version=seg_version)

    if args.rank == 0:
        pbar.close()
    
    if use_distributed:
        dist.destroy_process_group()

class WORKER_SLURM(object):
    def __init__(self, args):
        self.args = args
    def __call__(self):
        init_dist_node(self.args)
        WORKER(None, self.args)

def main(args_debug=None):
    args = parse_args(args_debug)
    args.port = random.randint(49152,65535)

    if hasattr(args, 'reset') and args.reset and os.path.exists(args.output_path):
        os.remove(args.output_path)
        print(f"Database reset: removed existing file {args.output_path}")
    if args.keep_branch:
        base_name = os.path.basename(args.output_path)
        if not str.endswith(base_name, '_keepBranch.db'):
            new_base_name = base_name.replace('.db', '_keepBranch.db')
            args.output_path = os.path.join(os.path.dirname(args.output_path), new_base_name)

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
        args.ngpus_per_node = len(args.gpus.split(',')) if hasattr(args, 'gpus') else 1
        if args.ngpus_per_node == 1:
            args.gpu = int(args.gpus.split(',')[0])
            WORKER(args.gpu, args)
        else:
            init_dist_node(args)
            mp.spawn(WORKER, args=(args,), nprocs=args.ngpus_per_node)

if __name__ == "__main__":
    # args_debug = [
    #     '-task', 'SegFiber_Debug',
    #     '-cfg', '/home/ryuuyou/E5/project/SegFiber_dev/eval/config/config.yaml',
    #     '-gpus', '0',
    #     '-reset'
    # ]
    args_debug = None
    main(args_debug)