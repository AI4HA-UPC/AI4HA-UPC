from typing import Union

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor


class GenericSegmentationDataset(Dataset):
    def __init__(self,
                 common_transform: Union[Compose, ToTensor] = ToTensor(),
                 transform: Union[Compose, ToTensor] = ToTensor(),
                 mask_transform: Union[Compose, ToTensor] = ToTensor()):

        # Applied first to both image and mask (for data augmentation purposes)
        # Leave empty if no data augmentation
        self.common_transform = common_transform
        # Applied afterwards to only image
        self.transform = transform
        # Applied afterwards to only mask
        self.mask_transform = mask_transform

        self.images = []
        self.masks = []
        self.cases = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        case = self.cases[idx]

        image = ToTensor()(Image.open(img_path))
        if mask_path is None:
            mask = torch.zeros((3, image.shape[1], image.shape[2]))
        else:
            mask = ToTensor()(Image.open(mask_path))

        batch = torch.stack((image, mask))
        batch = self.common_transform(batch)
        image, mask = batch

        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask, case
