import os

import torch

from ..core.registry import RUNTIME_REGISTRY
from .base import RuntimeContext, seed_process


@RUNTIME_REGISTRY.register("standalone")
class StandaloneRuntime:
    def launch(self, worker, config, paths):
        device_id = config["runtime"]["params"]["devices"][0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed = int(config["experiment"].get("seed", 31))
        seed_process(seed)
        context = RuntimeContext(
            rank=0,
            world_size=1,
            device=device,
            paths=paths,
            seed=seed,
        )
        return worker(config, context)
