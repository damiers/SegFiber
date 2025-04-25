import argparse

import submitit

import os, sys
import random
from pprint import pprint

from eval.seger import Seger

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# === arg praser ===
def parse_args(simulated_args=None):
    parser = argparse.ArgumentParser(description='Seger')

    # === GENERAL === #
    parser.add_argument('-task', type=str, default="seg_fiber",
                        help='task name')
    
    parser.add_argument('-reset', action='store_true',
                        help='Reset saved model logs and weights')
    
    parser.add_argument('-gpu', type=str, default="0",
                        help='GPU id')
    
    parser.add_argument('-cfg', type =str,
                        help='Configuration file')

    # === PATHS === #
    parser.add_argument('-input_path', type=str, default="data",
                        help='input path')
    
    parser.add_argument('-output_path', type=str, default="data",
                        help='output path')
    
    parser.add_argument('-mysql',   type=bool, default=False,
                        help='write results to mysql database')
    parser.add_argument('-db_url', type=str, default="localhost:3306",
                        help='url of the database')
    
    parser.add_argument('-db_username', type=str, default="root",
                        help='username of the database')
    
    parser.add_argument('-db_password', type=str, default="password",
                        help='password of the database')
    
    parser.add_argument('-db_name', type=str, default="smapleID",
                        help='name of the database')
    
    # === PARAMETERS === #
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
            cpus_per_task = 12,
            nodes = 1,
            timeout_min = 2800,
            slurm_partition = args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.task)
        trainer = worker_slurm(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")

    else:
        # init_node(args)
        worker(args)

# === worker ===
def worker(args):
    pprint(vars(args))
    # === PROCESS === #

    seger = Seger(ckpt_path=None, bg_thres=args.bg_thres, cuda_device_id=args.gpu)
    seger.process_whole(args.input_path, args.output_path, args.level, args.channel, chunk_size=args.chunk_size, splice=args.splice, roi=args.roi)

class worker_slurm(object):
    def __init__(self, args):
        self.args = args
    def __call__(self):
        # init_node(self.args)
        worker(self.args)
	
if __name__ == '__main__':
    # for test
    # simulated_args = ['-cfg', 'eval/config/config.yaml', '-gpu', '0']
    simulated_args = None
    main(simulated_args)
