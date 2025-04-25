from torch.utils.data import Dataset
import torch
import os

import tifffile as tiff
import numpy as np
from empatches import EMPatches

from tqdm import tqdm

def preprocess(img,percentiles=[0.01, 0.9999]):
    # input img [0,65535]
    # output img [0,1]
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr)) - 1
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])

    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    return img

class skelDataset(Dataset):
    def __init__(self, path, patch_size=64, overlap=0.0, use_preprocess=True) -> None:
        super().__init__()
        print(f'patch_size: {patch_size}')
        print(f'overlap: {overlap}')
        print(f'preprocess: {use_preprocess}')
        
        img_path = os.path.join(path, 'img')
        mask_path = os.path.join(path, 'mask')
        bg_path = os.path.join(path, 'img_bg')
        print(f'image path: {img_path}')
        print(f'mask path: {mask_path}')
        print(f'background path: {bg_path}')

        emp = EMPatches()
        # patches
        self.img_patches_list = []
        self.mask_patches_list = []

        # number of foreground points in each patches
        self.fgp_in_patches = []

        img_name_list = sorted(os.listdir(img_path))
        mask_name_list = sorted(os.listdir(mask_path))
        bg_name_list = os.listdir(bg_path)

        self.img_name_list = img_name_list
        self.mask_name_list = mask_name_list
        self.bg_name_list = bg_name_list

        # add fg img
        for index, (img_name, mask_name) in tqdm(enumerate(zip(img_name_list, mask_name_list))):
            img = tiff.imread(os.path.join(img_path, img_name))
            img = img.astype(np.float32)

            # preprocess image
            if use_preprocess:
                img = preprocess(img)

            mask = tiff.imread(os.path.join(mask_path, mask_name))
            mask = mask.astype(np.float32)

            # extract pathces
            img_patches, img_indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlap, stride=None, vox=True)
            mask_patches, mask_indices = emp.extract_patches(mask, patchsize=patch_size, overlap=overlap, stride=None, vox=True)
            
            # add patches
            self.img_patches_list = self.img_patches_list + img_patches
            self.mask_patches_list = self.mask_patches_list + mask_patches
        
        # indices of pathces
        self.patches_indices = mask_indices

        # add bg img
        for index, bg_name in tqdm(enumerate(bg_name_list)):
            bg = tiff.imread(os.path.join(bg_path, bg_name))
            bg = bg.astype(np.float32)

            # preprocess image
            if use_preprocess:
                bg = preprocess(bg)
            
            mask = np.zeros_like(bg, dtype=np.float32)

            # extract pathces
            bg_patches, bg_indices = emp.extract_patches(bg, patchsize=patch_size, overlap=overlap, stride=None, vox=True)
            mask_patches, mask_indices = emp.extract_patches(mask, patchsize=patch_size, overlap=overlap, stride=None, vox=True)

            # add patches
            self.img_patches_list = self.img_patches_list + bg_patches
            self.mask_patches_list = self.mask_patches_list + mask_patches

        # calculate foreground points
        for mp in self.mask_patches_list:
            self.fgp_in_patches.append(mp.sum())
        self.fgp_in_patches=np.asarray(self.fgp_in_patches)

        self.patches_per_volume = len(self.patches_indices)
        self.patches_num = len(self.img_patches_list)
        
        self.img_patches_list = np.asarray(self.img_patches_list)
        self.mask_patches_list = np.asarray(self.mask_patches_list)

    def __len__(self):
        return self.patches_num
        
    def __getitem__(self, index):
        img = self.img_patches_list[index]
        mask = self.mask_patches_list[index]
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask

def get_dataset(args):
    train_dataset = skelDataset(path=args.data, 
                                patch_size=args.patch_size, 
                                overlap=args.overlap)
    return train_dataset

if __name__ == "__main__":
    pass
