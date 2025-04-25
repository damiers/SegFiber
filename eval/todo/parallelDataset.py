from torch.utils.data import Dataset
import torch
import os

import numpy as np

from neurofly import image_reader

class parallelDataset(Dataset):
    def __init__(self, image_path:str, patch_size:int, slice_thickness:int, level:int=0, channel:int=0):
        super().__init__()
        self.IMAGE = image_reader(image_path)
        self.img_roi = self.IMAGE.rois[level]

        self.patch_size = patch_size
        self.slice_thickness = slice_thickness
        self.level = level,
        self.channel = channel

        self.border_width = 4
        self.PATCH_ROIS = self.patchify_without_splices()

    def patchify_without_splices(self):
        OFFSET, SIZE = self.img_roi[:3], self.img_roi[3:]
        ROIS = []
        X = list(range(OFFSET[0], OFFSET[0]+SIZE[0], self.patch_size))
        X.append(OFFSET[0] + SIZE[0])

        Y = list(range(OFFSET[1], OFFSET[1]+SIZE[1], self.patch_size))
        Y.append(OFFSET[1] + SIZE[1])

        if self.patch_size%self.slice_thickness == 0:
            Z = [z for z in range(OFFSET[2], OFFSET[2]+SIZE[2]) if z%self.slice_thickness==0 or z%self.patch_size==0]
            if OFFSET[2]%self.slice_thickness != 0:
                Z.insert(0, OFFSET[2])
        else:
            Z = list(range(OFFSET[2], OFFSET[2]+SIZE[2], self.patch_size))

        Z.append(OFFSET[2] + SIZE[2])

        for x1,x2 in zip(X[:-1],X[1:]):
            for y1,y2 in zip(Y[:-1],Y[1:]):
                for z1,z2 in zip(Z[:-1],Z[1:]):
                    ROIS.append(np.asarray([x1, y1, z1 , x2-x1, y2-y1, z2-z1]))
        return ROIS

    def __len__(self):
        return len(self.PATCH_ROIS)

    def __getitem__(self, idx):
        roi = self.PATCH_ROIS[idx]
        if (roi[3:] <= np.asarray([128, 128, 128])).all():
            img_patch = self.IMAGE.from_roi(roi, padding='reflect', level=self.level, channel=self.channel)
            re_batch = torch.tensor(False)
        else:
            roi[:3] = [i-self.border_width for i in roi[:3]]
            roi[3:] = [i+2*self.border_width for i in roi[3:]]
            img_patch = self.IMAGE.from_roi(roi, padding='reflect', level=self.level, channel=self.channel)
            re_batch = torch.tensor(True)
        img_patch = torch.from_numpy(img_patch).to(torch.float32)
        roi = torch.from_numpy(roi).to(torch.uint32)
        return img_patch, roi, re_batch
        
if __name__ == '__main__':
    img_path = 'test/data/test.tif'
    ds = parallelDataset(img_path, patch_size=300, slice_thickness=100, level=0, channel=0)
    print(len(ds))
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset=ds, 
        batch_size=1, 
        num_workers=2,
    )
    for i, (img_patch, roi, re_batch) in enumerate(loader):
        print(i, img_patch.shape, roi.shape, re_batch.dtype)