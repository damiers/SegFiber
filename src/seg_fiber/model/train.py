import os

from .config import prepare_run_paths, resolve_run_paths, save_config, to_plain_data
from .runtime import get_runtime
from .trainer import get_trainer


def train(config, reset=False, slurm=False):
    paths = resolve_run_paths(config)
    if slurm:
        if "SLURM_PROCID" not in os.environ:
            raise RuntimeError("--slurm must run inside an srun task.")
        if int(os.environ["SLURM_PROCID"]) == 0:
            save_config(config, paths.run / "config.yaml")
    else:
        prepare_run_paths(paths, reset=reset)
        save_config(config, paths.run / "config.yaml")

    run_config = to_plain_data(config)
    runtime = get_runtime(run_config)
    return runtime.launch(_run_worker, run_config, paths)


def _run_worker(config, context):
    return get_trainer(config).run(config, context)
