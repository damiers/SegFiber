from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    device: torch.device
    paths: object
    seed: int

    @property
    def is_main(self):
        return self.rank == 0

    def make_sampler(self, dataset, shuffle):
        if self.world_size == 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
        )

    def wrap_model(self, model):
        if self.world_size == 1:
            return model
        return DistributedDataParallel(model, device_ids=[self.device.index])

    def reduce_mean(self, value):
        if self.world_size == 1:
            return float(value)
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.world_size).item()


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
