import numpy as np
import torch

from ..infer import load_model


class TorchSegmenter:
    def __init__(self, config, checkpoint, background_threshold, device):
        self.model = load_model(config, checkpoint, device)
        self.background_threshold = background_threshold
        self.device = device

    def preprocess(self, image, percentiles=(0.1, 1.0)):
        image = np.clip(image, a_min=self.background_threshold, a_max=None)
        image = image - self.background_threshold
        values = np.sort(image.ravel())
        low = values[int(percentiles[0] * len(values))]
        high = values[int(percentiles[1] * len(values)) - 1]
        image = np.clip(image, low, high)
        minimum, maximum = image.min(), image.max()
        if maximum == minimum:
            return None
        image = ((image - minimum) / (maximum - minimum)).astype(np.float32)
        return torch.from_numpy(image)[None, None]

    @torch.no_grad()
    def get_mask(self, image, thres=0.5):
        return self.get_masks([image], thres=thres)[0]

    @torch.no_grad()
    def get_masks(self, images, thres=0.5):
        tensors = []
        indices = []
        masks = [np.zeros_like(image) for image in images]
        for index, image in enumerate(images):
            tensor = self.preprocess(image)
            if tensor is not None:
                tensors.append(tensor)
                indices.append(index)

        if not tensors:
            return masks

        probabilities = self.model(torch.cat(tensors).to(self.device))[:, 0].cpu()
        if thres is not None:
            probabilities = (probabilities >= thres).to(torch.float32)
        for index, probability in zip(indices, probabilities):
            masks[index] = probability.numpy()
        return masks
