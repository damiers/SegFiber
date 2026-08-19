import torch.optim as optim

from ..core.registry import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register("adam")
class AdamOptimizer(optim.Adam):
    @classmethod
    def init_from_config(cls, params, model, config):
        return cls(model.parameters(), **params)


@OPTIMIZER_REGISTRY.register("sgd")
class SGDOptimizer(optim.SGD):
    @classmethod
    def init_from_config(cls, params, model, config):
        return cls(model.parameters(), **params)


@OPTIMIZER_REGISTRY.register("adagrad")
class AdagradOptimizer(optim.Adagrad):
    @classmethod
    def init_from_config(cls, params, model, config):
        return cls(model.parameters(), **params)
