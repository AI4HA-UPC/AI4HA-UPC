import logging
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Grayscale, RandomHorizontalFlip, RandomRotation

import numpy as np
import cv2
import albumentations

log = logging.getLogger(__name__)


def squarify_mask(mask):
    nz = np.nonzero(mask)
    if nz[0].shape != 0:
        x1 = np.min(nz[0])
        x2 = np.max(nz[0])
        y1 = np.min(nz[1])
        y2 = np.max(nz[1])
        square_mask = np.zeros(mask.shape,dtype=np.uint8)
        square_mask[x1:x2, y1:y2] = 1
        return square_mask
    else:
        return mask


class SunDataset(Dataset):

    def __init__(self,
                 size=256,
                 data_csv='',
                 mask_csv='',
                 data_root='',
                 augmentation: bool = True):

        self.size = size
        self.data_csv = data_csv
        self.mask_csv = mask_csv
        self.data_root = data_root
        self.n_labels = 2
        self.squarify = True

        if augmentation:
            self.augmentation = albumentations.OneOf([
                albumentations.VerticalFlip(p=0.25),
                albumentations.HorizontalFlip(p=0.25),
                albumentations.RandomRotate90(p=0.25)
            ])
        else:
            self.augmentation = None

        if self.size is not None:
            self.image_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_CUBIC)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_NEAREST)
        self.preprocessor = albumentations.CenterCrop(height=self.size,
                                                      width=self.size)

        self.images = []
        self.masks = []

        with open(f"{self.data_csv}", "r") as f:
            self.images = f.read().splitlines()
        self._length = len(self.images)

        with open(f"{self.mask_csv}", "r") as f:
            self.masks = f.read().splitlines()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(self.data_root + '/' + img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        nomask = False
        if mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            nomask = True
        else:
            mask = Image.open(self.data_root + '/' + mask_path)
            # mask.save('1.jpg')
            mask = np.array(mask).astype(np.uint8)
            mask = np.array(mask // np.max(mask))
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]
            if self.squarify:
                mask = squarify_mask(mask)

        image = self.image_rescaler(image=image)["image"]
        mask = self.segmentation_rescaler(image=mask)["image"]
        processed = self.preprocessor(image=image, mask=mask)

        if self.augmentation:
            processed = self.augmentation(image=processed['image'],
                                          mask=processed['mask'])

        example = {}

        onehot = np.expand_dims(processed["mask"].astype(np.float32), axis=2)

        example["image"] = (processed["image"] / 127.5 - 1.0).astype(
                np.float32)
        try:
            example["segmentation"] = onehot
            example["imageseg"] = np.concatenate((example["image"], onehot),
                                                 axis=2)
        except:
            print(
                f"Error {img_path} {example['image'].shape} {mask_path} {onehot.shape}"
            )

        return example
