import numpy as np

from ..core.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("cosine")
class CosineScheduler:
    def __init__(
        self,
        optimizer,
        base_lr,
        final_lr,
        epochs,
        steps_per_epoch,
        warmup_epochs,
    ):
        self.optimizer = optimizer
        warmup_steps = warmup_epochs * steps_per_epoch
        warmup = np.linspace(0, base_lr, warmup_steps) if warmup_steps else []
        cosine_steps = epochs * steps_per_epoch - warmup_steps
        iterations = np.arange(cosine_steps)
        cosine = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * iterations / len(iterations))
        )
        self.values = np.concatenate((warmup, cosine))

    @classmethod
    def init_from_config(
        cls,
        params,
        optimizer,
        config,
        steps_per_epoch,
        world_size,
    ):
        trainer = config["trainer"]["params"]
        base_lr = config["optimizer"]["params"]["lr"]
        base_lr *= trainer["batch_size"] * world_size / params["reference_batch_size"]
        return cls(
            optimizer=optimizer,
            base_lr=base_lr,
            final_lr=params["final_lr"],
            epochs=trainer["epochs"],
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=params["warmup_epochs"],
        )

    def step(self, iteration):
        value = float(self.values[iteration])
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None
