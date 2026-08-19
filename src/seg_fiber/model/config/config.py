from dataclasses import dataclass
from pathlib import Path
import shutil

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq


@dataclass(frozen=True)
class RunPaths:
    run: Path
    checkpoints: Path
    logs: Path
    tensorboard: Path
    slurm: Path


def load_config(path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    return config


def save_config(config, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file)


def apply_overrides(config, overrides):
    for key_path, value in overrides:
        if value is None:
            continue
        keys = key_path.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown configuration key: {key_path}")
            target = target[key]
        key = keys[-1]
        if key not in target:
            raise KeyError(f"Unknown configuration key: {key_path}")
        target[key] = _preserve_sequence_style(target[key], value)
    return config


def parse_set_overrides(values):
    yaml = YAML(typ="safe")
    overrides = []
    for item in values or []:
        key, separator, raw_value = item.partition("=")
        if not separator or not key or not raw_value:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        overrides.append((key, yaml.load(raw_value)))
    return overrides


def parse_devices(devices):
    if isinstance(devices, (list, tuple)):
        return [int(device) for device in devices]
    return [int(device.strip()) for device in str(devices).split(",")]


def module_config(config, section_name, subsection=None):
    section = config.get(section_name)
    if subsection is not None:
        section = section.get(subsection) if isinstance(section, dict) else None
    label = f"{section_name}.{subsection}" if subsection else section_name
    if not isinstance(section, dict):
        raise ValueError(f"{label} must be a module configuration mapping.")
    name = section.get("name")
    params = section.get("params", {})
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.name must be a non-empty string.")
    if not isinstance(params, dict):
        raise ValueError(f"{label}.params must be a mapping.")
    return name, dict(params)


def resolve_run_paths(config):
    experiment = config["experiment"]
    root = Path(experiment["output_dir"]).expanduser()
    name = experiment["name"]
    return RunPaths(
        run=root / name,
        checkpoints=root / "weights" / name,
        logs=root / "logs" / name,
        tensorboard=root / "logs" / name,
        slurm=root / "slurm",
    )


def prepare_run_paths(paths, reset=False):
    if reset:
        for path in (paths.run, paths.checkpoints, paths.logs):
            if path.exists():
                shutil.rmtree(path)
    for path in (
        paths.run,
        paths.checkpoints,
        paths.logs,
        paths.tensorboard,
        paths.slurm,
    ):
        path.mkdir(parents=True, exist_ok=True)


def to_plain_data(value):
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def _preserve_sequence_style(current, value):
    if not isinstance(value, (list, tuple)):
        return value
    sequence = CommentedSeq(value)
    if isinstance(current, CommentedSeq) and current.fa.flow_style():
        sequence.fa.set_flow_style()
    return sequence
