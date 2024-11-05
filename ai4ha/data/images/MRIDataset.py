import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):

    def __init__(self, size=256, data_root='/NautilusMRI/', dataset='train'):
        self.images = [
            f for f in os.listdir(f'{data_root}/{dataset}/')
            if f.endswith('.npz')
        ]
        self.size = size
        self.data_root = data_root
        self.dataset = dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(
            f'{self.data_root}/{self.dataset}/{self.images[idx]}')['data']
        image = torch.tensor(image).float()
        cls = int(self.images[idx].split('_')[0]) - 1
        sample = {'image': image.unsqueeze(2), 'cls': cls}
        return sample
