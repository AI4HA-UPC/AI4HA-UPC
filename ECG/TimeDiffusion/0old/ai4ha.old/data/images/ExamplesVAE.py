import os
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import rand


class ExamplesVAE(Dataset):

    def __init__(self,
                 model='AEKL',
                 size=256,
                 encoder='AllAEKL64',
                 root='/gpfs/projects/bsc70/bsc70642/Data/LDPolyp',
                 normalization=1.,
                 segmentation=False,
                 augmentation=False,
                 augprob=0.5):
        base_path = f'{root}/{model}/{size}/{encoder}/'
        if augmentation:
            base_path += 'aug/'
            self.images = [
                os.path.join(os.path.join(base_path, 'images'), d)
                for d in sorted(os.listdir(os.path.join(base_path, 'images')))
                if 'orig' in d
            ]
            if segmentation:
                self.segmentations = [
                    os.path.join(os.path.join(base_path, 'masks'), d)
                    for d in sorted(
                        os.listdir(os.path.join(base_path, 'masks')))
                    if 'orig' in d
                ]
        else:
            self.images = [
                os.path.join(os.path.join(base_path, 'images'), d)
                for d in sorted(os.listdir(os.path.join(base_path, 'images')))
            ]

            if segmentation:
                self.segmentations = [
                    os.path.join(os.path.join(base_path, 'masks'), d)
                    for d in sorted(
                        os.listdir(os.path.join(base_path, 'masks')))
                ]

        self.normalization = normalization
        self.seg = segmentation
        self.augprob = augprob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        ntrans = 5
        augdiv = self.augprob / ntrans
        ltrans = ['vflip', 'hflip', 'rot90', 'rot180', 'rot270', 'orig']
        vtrans = []
        for i in range(ntrans):
            vtrans.append([augdiv * (i + 1), ltrans[i]])
        vtrans.append([1, 'orig'])

        rep = 'orig'
        for i in range(ntrans + 1):
            if rand(1)[0] < vtrans[i][0]:
                rep = vtrans[i][1]
                break

        img_path = self.images[idx].replace('orig', rep)
        # print(img_path)
        if self.seg:
            image = torch.tensor(
                np.load(img_path) / (1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace('orig', rep)
            onehot = torch.tensor(np.load(mask_path))
            # s_onehot = ((onehot/onehot.max())*2)-1
            example = {
                "image": image,
                "segmentation": onehot,
                # "imageseg": np.concatenate((image, s_onehot), axis=0)
            }
        else:
            example = {
                "image":
                torch.tensor(np.load(img_path) / (1.0 * self.normalization))
            }
        return example
