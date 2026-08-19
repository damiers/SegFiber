import numpy as np
from torch.utils.data import Dataset

from ..core.registry import DATASET_REGISTRY
from ..utils.image_reader import ImageReader


@DATASET_REGISTRY.register("brain_z_slabs")
class BrainZSlabDataset(Dataset):
    def __init__(
        self,
        image_path,
        patch_size,
        slab_thickness,
        level=0,
        channel=0,
        roi=None,
    ):
        self.image = ImageReader(image_path)
        self.image_roi = np.asarray(
            roi if roi is not None else self.image.rois[level]
        )
        self.patch_size = patch_size
        self.slab_thickness = slab_thickness
        self.level = level
        self.channel = channel
        self.border_width = 4
        self.slab_count = int(self.image_roi[5] // slab_thickness)
        if self.image_roi[5] % slab_thickness:
            raise ValueError("The ROI depth must be divisible by slab_thickness.")
        self.patch_specs = self._make_patch_specs()

    @classmethod
    def init_from_config(cls, params, config):
        return cls(**params)

    def _make_patch_specs(self):
        x0, y0, z0, width, height, _ = map(int, self.image_roi)
        xs = list(range(x0, x0 + width, self.patch_size)) + [x0 + width]
        ys = list(range(y0, y0 + height, self.patch_size)) + [y0 + height]
        specs = []
        for slab_index in range(self.slab_count):
            z = z0 + slab_index * self.slab_thickness
            patch_index = 0
            for x1, x2 in zip(xs[:-1], xs[1:]):
                for y1, y2 in zip(ys[:-1], ys[1:]):
                    roi = np.asarray(
                        [x1, y1, z, x2 - x1, y2 - y1, self.slab_thickness]
                    )
                    specs.append((slab_index + 1, patch_index, roi))
                    patch_index += 1
        return specs

    def __len__(self):
        return len(self.patch_specs)

    def __getitem__(self, index):
        slab_number, slab_patch_index, roi = self.patch_specs[index]
        roi = roi.copy()
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
            "slab_number": slab_number,
            "slab_patch_index": slab_patch_index,
            "image": image,
            "offset": offset,
            "rebatch": rebatch,
        }


def collate_brain_z_slabs(batch):
    return batch[0]
