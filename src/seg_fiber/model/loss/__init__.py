from ..config import module_config
from ..core.registry import LOSS_REGISTRY, discover


discover(__name__, "_loss")


def get_loss(config):
    name, params = module_config(config, "loss")
    return LOSS_REGISTRY.get(name).init_from_config(params, config)


def list_losses():
    return LOSS_REGISTRY.available()
