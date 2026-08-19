from ..config import module_config
from ..core.registry import ARCH_REGISTRY, discover


discover(__name__, "_arch")


def get_model(config):
    name, params = module_config(config, "arch")
    return ARCH_REGISTRY.get(name).init_from_config(params, config)


def list_models():
    return ARCH_REGISTRY.available()
