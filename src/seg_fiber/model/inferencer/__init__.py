from ..config import module_config
from ..core.registry import INFERENCER_REGISTRY, discover


discover(__name__, "_inferencer")


def get_inferencer(config):
    name, params = module_config(config, "inference")
    return INFERENCER_REGISTRY.get(name)(config, params)


def list_inferencers():
    return INFERENCER_REGISTRY.available()
