import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIColDataset(Dataset):

    def __init__(self,
                 size=256,
                 data_root=None,
                 datasets=None,
                 partition='train'):
        self.size = size
        self.data_root = data_root
        self.dataset = datasets
        self.partition = partition
        self.images = []
        for dset in datasets:
            self.images += [
                f'{self.data_root}/{dset}/{partition}/{f}'
                for f in os.listdir(f'{self.data_root}/{dset}/{partition}')
                if f.endswith('.npz')
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])['data']
        image = torch.tensor(image).float()
        sample = {'image': image.unsqueeze(2)}
        return sample
