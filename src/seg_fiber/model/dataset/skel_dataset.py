from pathlib import Path

from empatches import EMPatches
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..core.registry import DATASET_REGISTRY


def preprocess_image(image, percentiles=(0.01, 0.9999)):
    values = np.sort(image.ravel())
    low = values[int(percentiles[0] * len(values))]
    high = values[int(percentiles[1] * len(values)) - 1]
    clipped = np.clip(image, low, high)
    minimum, maximum = clipped.min(), clipped.max()
    if maximum == minimum:
        return np.zeros_like(clipped, dtype=np.float32)
    return (clipped - minimum) / (maximum - minimum)


@DATASET_REGISTRY.register("skeleton")
class SkeletonDataset(Dataset):
    def __init__(self, path, patch_size=64, overlap=0.0, preprocess=True):
        root = Path(path)
        image_dir = root / "img"
        mask_dir = root / "mask"
        background_dir = root / "img_bg"
        patcher = EMPatches()
        self.images = []
        self.masks = []

        image_names = sorted(image_dir.iterdir(), key=lambda path: path.stem)
        mask_names = sorted(mask_dir.iterdir(), key=lambda path: path.stem)
        if [path.stem for path in image_names] != [path.stem for path in mask_names]:
            raise ValueError("Image and mask filenames must match.")
        for image_path, mask_path in tqdm(zip(image_names, mask_names)):
            image = tifffile.imread(image_path).astype(np.float32)
            if preprocess:
                image = preprocess_image(image)
            mask = tifffile.imread(mask_path).astype(np.float32)
            image_patches, _ = patcher.extract_patches(
                image, patchsize=patch_size, overlap=overlap, stride=None, vox=True
            )
            mask_patches, _ = patcher.extract_patches(
                mask, patchsize=patch_size, overlap=overlap, stride=None, vox=True
            )
            self.images.extend(image_patches)
            self.masks.extend(mask_patches)

        if background_dir.is_dir():
            for image_path in tqdm(sorted(background_dir.iterdir())):
                image = tifffile.imread(image_path).astype(np.float32)
                if preprocess:
                    image = preprocess_image(image)
                mask = np.zeros_like(image, dtype=np.float32)
                image_patches, _ = patcher.extract_patches(
                    image, patchsize=patch_size, overlap=overlap, stride=None, vox=True
                )
                mask_patches, _ = patcher.extract_patches(
                    mask, patchsize=patch_size, overlap=overlap, stride=None, vox=True
                )
                self.images.extend(image_patches)
                self.masks.extend(mask_patches)

        self.images = np.asarray(self.images)
        self.masks = np.asarray(self.masks)

    @classmethod
    def init_from_config(cls, params, config):
        return cls(**params)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index][None])
        mask = torch.from_numpy(self.masks[index][None])
        return image, mask
