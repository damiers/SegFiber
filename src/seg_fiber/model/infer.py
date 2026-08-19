from dataclasses import dataclass

from ..ckpt import load_model_state
from .arch import get_model
from .config import resolve_run_paths
from .runtime import get_runtime


def load_model(config, checkpoint, device):
    model = get_model(config).to(device)
    model.load_state_dict(load_model_state(checkpoint, map_location=device), strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class _InferenceWorker:
    reset: bool

    def __call__(self, config, context):
        from .inferencer import get_inferencer

        return get_inferencer(config).run(context, reset=self.reset)


def infer(config, reset=False):
    runtime = get_runtime(config)
    paths = resolve_run_paths(config)
    return runtime.launch(_InferenceWorker(reset), config, paths)
