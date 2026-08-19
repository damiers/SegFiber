from importlib.resources import files
from pathlib import Path

import torch


def load_checkpoint(source, map_location="cpu"):
    path = Path(source).expanduser()
    if path.is_file():
        return torch.load(path, map_location=map_location, weights_only=True)
    resource = files(__package__).joinpath(source)
    if not resource.is_file():
        raise FileNotFoundError(source)
    with resource.open("rb") as file:
        return torch.load(file, map_location=map_location, weights_only=True)


def load_model_state(source, map_location="cpu"):
    checkpoint = load_checkpoint(source, map_location=map_location)
    return checkpoint["model"] if "model" in checkpoint else checkpoint


__all__ = ["load_checkpoint", "load_model_state"]
