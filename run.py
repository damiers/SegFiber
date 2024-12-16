import torch

import argparse

import signal
import subprocess
import submitit

import os, sys
import random
from pathlib import Path
from pprint import pprint

from seg_fiber import Seger
from seg_fiber import segs2db


# === distributed helper function ===
def init_gpu(gpu, args):
    if args.slurm:
        job_env = submitit.JobEnvironment()
        args.slurm_log_dir = Path(str(args.slurm_log_dir).replace("%j", str(job_env.job_id)))
        check_dir(args.slurm_log_dir)
        args.gpu = job_env.local_rank
    else:
        args.gpu = gpu

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def init_node(args):
    if 'SLURM_JOB_ID' in os.environ:
        print('cuda available: ', torch.cuda.is_available())
        print('gpu device: ', torch.cuda.is_available())

        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # find a common host name on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.dist_url = f'tcp://{host_name}:{args.port}'
            
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.dist_url = f'tcp://localhost:{args.port}'

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === worker ===
def worker(gpu, args):
    pprint(vars(args))

    # === SET ENV === #
    init_gpu(gpu, args)
    
    # === PROCESS === #
    seger = Seger(ckpt_path=None, bg_thres=args.bg_thres, cuda_device_id=args.gpu)
    segs = seger.process_whole(args.input_path, args.level, args.channel, chunk_size=args.chunk_size, splice=args.splice, roi=args.roi)
    segs2db(segs, args.output_path)
    seger.connect_segs(args.output_path)

class slurm_worker(object):
    def __init__(self, args):
        self.args = args
    def __call__(self):
        init_node(self.args)
        worker(None, self.args)

# === arg praser ===
def parse_args(simulated_args=None):
    parser = argparse.ArgumentParser(description='Seger')

    # === GENERAL === #
    parser.add_argument('-task', type=str, default="seg_fiber",
                                            help='task name')
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-gpu', type=str, default="0",
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
    
    parser.add_argument('-chunk_size', type=int, default=0,
                                            help='the size of the cube accepted by the segmentation model')
                                            
    parser.add_argument('-splice', type=int, default=300,
                                            help='thickness of a brain slice')
    
    parser.add_argument('-roi', type=int, nargs='+', default=None,
                                            help='roi')

    # === SLURM === #
    parser.add_argument('-slurm', action='store_true', default=False,
                                            help='Submit with slurm')
    parser.add_argument('-slurm_nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-slurm_partition', type=str, default = "general",
                                            help='slurm partition')
    parser.add_argument('-slurm_timeout', type=int, default = 2800,
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')

    if simulated_args:
        args = parser.parse_args(simulated_args)
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

# === main ===
def main(simulated_args=None):
    args = parse_args(simulated_args)
    args.port = random.randint(49152,65535)

    pprint(vars(args))
    
    if args.slurm:
        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.slurm_log_dir = os.path.join(os.path.dirname(args.output_path), 'slurm_log/%j')
        executor = submitit.AutoExecutor(folder=args.slurm_log_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb = 12*1,
            gpus_per_node = 1,
            tasks_per_node = 1,
            cpus_per_task = 2,
            nodes = 1,
            timeout_min = 2800,
            slurm_partition = args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.task)
        trainer = slurm_worker(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")

    else:
        init_node(args)
        worker(args.gpu, args)
	
if __name__ == '__main__':
    # for test
    # simulated_args = ['-cfg', 'config.yaml', '-gpu', '0']
    simulated_args = None
    main(simulated_args)
