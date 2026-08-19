import numpy as np
from torch.utils.data import Dataset

from ..core.registry import DATASET_REGISTRY
from ..utils.image_reader import ImageReader


@DATASET_REGISTRY.register("whole_brain")
class WholeBrainDataset(Dataset):
    def __init__(
        self,
        image_path,
        patch_size,
        slice_thickness,
        level=0,
        channel=0,
        roi=None,
    ):
        self.image = ImageReader(image_path)
        self.image_roi = roi if roi is not None else self.image.rois[level]
        self.patch_size = patch_size
        self.slice_thickness = slice_thickness
        self.level = level
        self.channel = channel
        self.border_width = 4
        self.patch_rois = self._make_rois()

    @classmethod
    def init_from_config(cls, params, config):
        return cls(**params)

    def _make_rois(self):
        offset, size = self.image_roi[:3], self.image_roi[3:]
        xs = list(range(offset[0], offset[0] + size[0], self.patch_size))
        ys = list(range(offset[1], offset[1] + size[1], self.patch_size))
        xs.append(offset[0] + size[0])
        ys.append(offset[1] + size[1])

        if self.patch_size % self.slice_thickness == 0:
            zs = [
                z
                for z in range(offset[2], offset[2] + size[2])
                if z % self.slice_thickness == 0 or z % self.patch_size == 0
            ]
            if offset[2] % self.slice_thickness:
                zs.insert(0, offset[2])
        else:
            zs = list(range(offset[2], offset[2] + size[2], self.patch_size))
        zs.append(offset[2] + size[2])

        return [
            np.asarray([x1, y1, z1, x2 - x1, y2 - y1, z2 - z1])
            for x1, x2 in zip(xs[:-1], xs[1:])
            for y1, y2 in zip(ys[:-1], ys[1:])
            for z1, z2 in zip(zs[:-1], zs[1:])
        ]

    def __len__(self):
        return len(self.patch_rois)

    def __getitem__(self, index):
        roi = self.patch_rois[index].copy()
        if (roi[3:] <= np.asarray([128, 128, 128])).all():
            rebatch = False
            offset = roi[:3]
        else:
            roi[:3] -= self.border_width
            roi[3:] += 2 * self.border_width
            rebatch = True
            offset = roi[:3] + self.border_width
        image = self.image.from_roi(
            roi,
            padding="reflect",
            level=self.level,
            channel=self.channel,
        ).astype(np.float32)
        return {
            "index": index,
            "image": image,
            "offset": offset,
            "rebatch": rebatch,
        }


def collate_whole_brain(batch):
    return batch[0]
