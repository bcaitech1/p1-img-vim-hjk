import os
import numpy as np
import pandas as pd
from glob import glob

import cv2
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


class MaskDataset(Dataset):
    def __init__(self, label_path='../label/Base/whole_label.csv', transform=None):
        self.label_path = label_path        
        self.transform = transform
        self.label_df = pd.read_csv(os.path.join(self.label_path))
        self.cls = self.label_df['class']
        self.age = self.label_df['age']
        self.img_path = self.label_df['image_path']

    def __len__(self):
        return len(self.label_df)
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.label_df.loc[idx, 'image_path']))
        
        label = int(self.label_df.loc[idx, 'class'])
        mask = int(self.label_df.loc[idx, 'mask'])
        gender = int(self.label_df.loc[idx, 'gender'])
        age = int(self.label_df.loc[idx, 'age'])
        
        if self.transform:
            img = self.transform(image=np.array(img))['image']

        return img, label, mask, gender, age
    
class Subset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return self.transform(im), labels

    def __len__(self):
        return len(self.indices)

class AugMix(ImageOnlyTransform):
    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")