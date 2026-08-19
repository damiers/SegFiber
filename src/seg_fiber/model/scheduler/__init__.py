from ..config import module_config
from ..core.registry import SCHEDULER_REGISTRY, discover


discover(__name__, "_scheduler")


def get_scheduler(config, optimizer, steps_per_epoch, world_size):
    name, params = module_config(config, "scheduler")
    return SCHEDULER_REGISTRY.get(name).init_from_config(
        params,
        optimizer,
        config,
        steps_per_epoch,
        world_size,
    )


def list_schedulers():
    return SCHEDULER_REGISTRY.available()
