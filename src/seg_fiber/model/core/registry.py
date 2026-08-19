import importlib
import pkgutil


class Registry:
    def __init__(self, label):
        self.label = label
        self._items = {}

    def register(self, name):
        def decorator(obj):
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.label}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name):
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"{name!r} is not registered in {self.label}. Available: {available}"
            )
        return self._items[name]

    def available(self):
        return sorted(self._items)


def discover(package_name, suffix):
    package = importlib.import_module(package_name)
    names = sorted(
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if info.name.endswith(suffix)
    )
    for name in names:
        importlib.import_module(f"{package_name}.{name}")


ARCH_REGISTRY = Registry("architecture")
DATASET_REGISTRY = Registry("dataset")
LOSS_REGISTRY = Registry("loss")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
TRAINER_REGISTRY = Registry("trainer")
INFERENCER_REGISTRY = Registry("inferencer")
RUNTIME_REGISTRY = Registry("runtime")
