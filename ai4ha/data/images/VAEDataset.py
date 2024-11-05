import os
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import rand


class VAEDataset(Dataset):

    def __init__(self,
                 model='AEKL',
                 size=256,
                 encoder='AllAEKL64',
                 data_root='/',
                 datasets='LDPolyp1DM',
                 normalization=1.,
                 segmentation=False,
                 augmentation=False,
                 transform=['vflip', 'hflip', 'rot90', 'rot180', 'rot270'],
                 augprob=0.5):

        if isinstance(datasets, str):
            datasets = [datasets]

        self.images = []
        self.segmentations = []

        self.normalization = normalization
        self.seg = segmentation
        self.augprob = augprob
        self.transform = transform

        base_paths = []
        for r in datasets:
            base_path = f'{data_root}/{r}/{model}/{size}/{encoder}/'
            base_paths.append(base_path)

        for base_path in base_paths:
            if augmentation:
                base_path += 'aug/'
                self.images.extend([
                    os.path.join(os.path.join(base_path, 'images'), d)
                    for d in sorted(
                        os.listdir(os.path.join(base_path, 'images')))
                    if 'orig' in d
                ])
                if segmentation:
                    self.segmentations.extend([
                        os.path.join(os.path.join(base_path, 'masks'), d)
                        for d in sorted(
                            os.listdir(os.path.join(base_path, 'masks')))
                        if 'orig' in d
                    ])
            else:
                self.images.extend([
                    os.path.join(os.path.join(base_path, 'images'), d)
                    for d in sorted(
                        os.listdir(os.path.join(base_path, 'images')))
                ])

                if segmentation:
                    self.segmentations.extend([
                        os.path.join(os.path.join(base_path, 'masks'), d)
                        for d in sorted(
                            os.listdir(os.path.join(base_path, 'masks')))
                    ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ntrans = len(self.transform)
        augdiv = self.augprob / ntrans

        vtrans = []
        for i, t in enumerate(self.transform):
            vtrans.append([augdiv * (i + 1), t])
        vtrans.append([1, 'orig'])

        rep = 'orig'
        if self.augprob > 0:
            for i in range(ntrans + 1):
                if rand(1)[0] < vtrans[i][0]:
                    rep = vtrans[i][1]
                    break

        img_path = self.images[idx].replace('orig', rep)

        if self.seg:
            image = torch.tensor(
                np.load(img_path) / (1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace('orig', rep)
            onehot = torch.tensor(np.load(mask_path).astype(np.float32))
            example = {
                "image": image,
                "segmentation": onehot,
            }
        else:
            example = {
                "image":
                torch.tensor(np.load(img_path) / (1.0 * self.normalization))
            }
        return example
