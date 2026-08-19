import os
import random
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..core.registry import RUNTIME_REGISTRY
from .base import RuntimeContext, seed_process


@RUNTIME_REGISTRY.register("ddp")
class DDPRuntime:
    def launch(self, worker, config, paths):
        if "SLURM_PROCID" in os.environ:
            return self._launch_slurm(worker, config, paths)
        return self._launch_local(worker, config, paths)

    def _launch_local(self, worker, config, paths):
        devices = config["runtime"]["params"]["devices"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", _find_free_port())
        if not torch.cuda.is_available():
            raise RuntimeError("The DDP runtime requires CUDA devices.")
        return mp.spawn(
            _worker,
            args=(len(devices), worker, config, paths),
            nprocs=len(devices),
            join=True,
        )

    def _launch_slurm(self, worker, config, paths):
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        global_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        if "MASTER_ADDR" not in os.environ:
            if int(os.environ.get("SLURM_NNODES", "1")) > 1:
                raise RuntimeError("MASTER_ADDR is required for multi-node Slurm.")
            os.environ["MASTER_ADDR"] = "localhost"
        os.environ.setdefault("MASTER_PORT", _slurm_master_port())
        if not torch.cuda.is_available():
            raise RuntimeError("The DDP runtime requires CUDA devices.")
        device_count = torch.cuda.device_count()
        device_index = 0 if device_count == 1 else local_rank
        return _worker(
            device_index,
            world_size,
            worker,
            config,
            paths,
            global_rank=global_rank,
        )


def _worker(local_rank, world_size, worker, config, paths, global_rank=None):
    global_rank = local_rank if global_rank is None else global_rank
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed = int(config["experiment"].get("seed", 31))
    seed_process(seed + global_rank)
    context = RuntimeContext(
        rank=global_rank,
        world_size=world_size,
        device=device,
        paths=paths,
        seed=seed,
    )
    try:
        return worker(config, context)
    finally:
        dist.destroy_process_group()


def _find_free_port(low=20000, high=29999, tries=100):
    for _ in range(tries):
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
            return str(port)
    raise RuntimeError("Could not find a free DDP port.")


def _slurm_master_port():
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    return str(20000 + int(job_id) % 10000) if job_id and job_id.isdigit() else "29500"
