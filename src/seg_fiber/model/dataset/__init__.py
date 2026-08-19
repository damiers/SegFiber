from ..config import module_config
from ..core.registry import DATASET_REGISTRY, discover


discover(__name__, "_dataset")


def get_dataset(config, mode="train"):
    name, params = module_config(config, "dataset", subsection=mode)
    return DATASET_REGISTRY.get(name).init_from_config(params, config)


def list_datasets():
    return DATASET_REGISTRY.available()
