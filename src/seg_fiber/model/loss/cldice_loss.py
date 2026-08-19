import torch
from torch import nn

from ..core.registry import LOSS_REGISTRY


def soft_skeletonize(inputs, iterations):
    x = inputs
    for _ in range(iterations):
        p1 = -torch.nn.functional.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
        p2 = -torch.nn.functional.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
        p3 = -torch.nn.functional.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
        minimum = torch.min(torch.min(p1, p2), p3)
        contour = torch.nn.functional.relu(
            torch.nn.functional.max_pool3d(minimum, 3, 1, 1) - minimum
        )
        x = torch.nn.functional.relu(x - contour)
    return x


def positive_intersection(center_line, vessel):
    center_line = center_line.view(*center_line.shape[:2], -1)
    vessel = vessel.view(*vessel.shape[:2], -1)
    intersection = (center_line * vessel).sum(-1)
    return (intersection.sum(0) + 1e-12) / (
        center_line.sum(-1).sum(0) + 1e-12
    )


@LOSS_REGISTRY.register("cldice")
class ClDiceLoss(nn.Module):
    def __init__(self, skeleton_iterations=5):
        super().__init__()
        self.skeleton_iterations = skeleton_iterations

    @classmethod
    def init_from_config(cls, params, config):
        return cls(**params)

    def forward(self, prediction, target):
        target_skeleton = soft_skeletonize(target, self.skeleton_iterations)
        prediction_skeleton = soft_skeletonize(
            prediction, self.skeleton_iterations
        )
        recall = positive_intersection(target_skeleton, prediction)[0]
        precision = positive_intersection(prediction_skeleton, target)[0]
        return 1 - (2 * recall * precision) / (recall + precision)
