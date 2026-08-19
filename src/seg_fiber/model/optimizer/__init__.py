from ..config import module_config
from ..core.registry import OPTIMIZER_REGISTRY, discover


discover(__name__, "_optimizer")


def get_optimizer(config, model):
    name, params = module_config(config, "optimizer")
    return OPTIMIZER_REGISTRY.get(name).init_from_config(params, model, config)


def list_optimizers():
    return OPTIMIZER_REGISTRY.available()
