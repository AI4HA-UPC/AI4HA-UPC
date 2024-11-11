import logging
import os
import random

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

log = logging.getLogger(__name__)


DATASETS = [
    ['TR512', 1, 10, 256],
    ['TR512', 2, 10, 256],
    ['TR512', 3, 15, 256],
    ['TR512aug', 1, 10, 256],
    ['TR1024aug', 1, 10, 256],
    ['TR1024aug', 1, 10, 256], 
    ['DDPM256', 500, 0, 256], 
    ['DDPM256', 1000, 0, 256], 
    ['DDPM256SNR', 100, 0, 256], 
    ['DDPM256SNR', 500, 0, 256] 
]


class LDPolypGenDataset(Dataset):
    """_LDPOLYP generated datasets loader_
    """

    def __init__(self, trans='TR512', temp=1, topk=10, size=256,
                 normalize=False):
        """_Initialization method_

        Args:
            trans (str, optional): _Transformer model_. Defaults to 'TR512'.
            temp (int, optional): _Temperature sampling_. Defaults to 1.
            topk (int, optional): _Top K sampling_. Defaults to 10.
            size (int, optional): _image size_. Defaults to 256.
        """

        if [trans, temp, topk, size] not in DATASETS:
            raise NameError("LDPOLYP: Unknown dataset")

        if 'TR' in trans:
            base_path = f'/home/bejar/bsc/Data/LDPolypGen/TT{size}/{trans}/T{temp}_K{topk}'
        elif 'DD' in trans:
            base_path = f'/home/bejar/bsc/Data/LDPolypGen/DD{size}/{trans}/S{temp}'

        samples_root = 'samples'
        masks_root = f'segmentations'

        self.images = [os.path.join(os.path.join(base_path, samples_root), d)
                       for d in sorted(os.listdir(os.path.join(base_path, samples_root)))]
        self.masks = [os.path.join(os.path.join(base_path, masks_root), d)
                      for d in sorted(os.listdir(os.path.join(base_path, masks_root)))]
        
        if 'TR' in trans:
            self.data_id = [f'TT{size}-{trans}-T{temp}-k{topk}' for _ in self.images]
        elif 'DD' in trans:
            self.data_id = [f'DD{size}-{trans}-S{temp}' for _ in self.images]

        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        try:
            image = Image.open(img_path)
            if self.normalize:
                image = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)
            else:
                image = Compose([ToTensor()])(image)

            mask = np.asarray(Image.open(mask_path), dtype=np.int64)
            mask = torch.from_numpy(mask[:, :, 1]//255).float()
            mask = mask.unsqueeze(0)
        except:
            image = None
            mask = None

        return image, mask, img_path


class SUNcolonGenDataset(Dataset):
    """_SUN generated datasets loader_
    """
    def __init__(self, trans='DDPM256SNR',  temp=500, size=256,
                 normalize=False):
        """_Initialization method_

        Args:
            trans (str, optional): _Transformer model_. Defaults to 'TR512'.
            temp (int, optional): _Temperature sampling_. Defaults to 1.
            topk (int, optional): _Top K sampling_. Defaults to 10.
            size (int, optional): _image size_. Defaults to 256.
        """
        base_path = f'/home/bejar/bsc/Data/SUNcolonGen/DD{size}/{trans}/S{temp}'

        samples_root = 'samples'
        masks_root = f'segmentations'

        self.images = [os.path.join(os.path.join(base_path, samples_root), d)
                       for d in sorted(os.listdir(os.path.join(base_path, samples_root)))]
        self.masks = [os.path.join(os.path.join(base_path, masks_root), d)
                      for d in sorted(os.listdir(os.path.join(base_path, masks_root)))]

        self.data_id = [f'DD{size}-{trans}-S{temp}' for _ in self.images]
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        try:
            image = Image.open(img_path)
            if self.normalize:
                image = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)
            else:
                image = Compose([ToTensor()])(image)

            mask = np.asarray(Image.open(mask_path), dtype=np.int64)
            mask = torch.from_numpy(mask[:, :, 1]//255).float()
            mask = mask.unsqueeze(0)
        except:
            image = None
            mask = None

        return image, mask, img_path
    

class ProstateGenDataset(Dataset):
    """_Prostate generated datasets loader_
    """
    def __init__(self, name='DDPM256SNR', temp=500, size=256,
                 normalize=False):
        """_Initialization method_

        Args:
            trans (str, optional): _Transformer model_. Defaults to 'TR512'.
            temp (int, optional): _Temperature sampling_. Defaults to 1.
            topk (int, optional): _Top K sampling_. Defaults to 10.
            size (int, optional): _image size_. Defaults to 256.
        """
        self.base_path = f'/gpfs/projects/bsc70/bsc70174/PANDA_code/logs/{name}/gsamples/{temp}'

        samples_root = 'sample'
        masks_root = f'mask'

        paths = os.listdir(self.base_path)
        self.images = [p for p in paths if p.endswith(f'{samples_root}.jpg')]
        self.masks = [p for p in paths if p.endswith(f'{masks_root}.png')]

        self.data_id = [f'DD{size}-{name}-S{temp}' for _ in self.images]
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(os.path.join(self.base_path, img_path))
        if self.normalize:
            image = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)
        else:
            image = Compose([ToTensor()])(image)

        mask = np.asarray(Image.open(os.path.join(self.base_path, mask_path)), dtype=np.int64)
        mask = torch.from_numpy((mask[:, :, 0]/255)*5).int()
        mask = mask.unsqueeze(0)

        return image*255, mask, img_path
