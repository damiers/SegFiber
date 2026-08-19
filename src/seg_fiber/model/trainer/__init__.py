from ..config import module_config
from ..core.registry import TRAINER_REGISTRY, discover


discover(__name__, "_trainer")


def get_trainer(config):
    name, _ = module_config(config, "trainer")
    return TRAINER_REGISTRY.get(name)()


def list_trainers():
    return TRAINER_REGISTRY.available()
