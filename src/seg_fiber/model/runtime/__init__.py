from ..config import module_config
from ..core.registry import RUNTIME_REGISTRY, discover


discover(__name__, "_runtime")


def get_runtime(config):
    name, _ = module_config(config, "runtime")
    return RUNTIME_REGISTRY.get(name)()


def list_runtimes():
    return RUNTIME_REGISTRY.available()
