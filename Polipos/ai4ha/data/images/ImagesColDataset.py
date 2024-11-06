import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch


class ImagesColDataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 data_csv,
                 data_root,
                 size=None,
                 center_crop=True,
                 interpolation="bicubic",
                 augmentation=False,
                 fld=False):
        self.data_csv = data_csv
        self.data_root = dataset_dir
        self.root = data_root
        self.images = []
        self.size = size
        self.center_crop = center_crop
        self.interpolation = interpolation
        self.fld = fld

        for droot, dcsv in zip(self.data_root, self.data_csv):
            with open(f"{self.root}/{dcsv}", "r") as f:
                image_paths = f.read().splitlines()

            self.images = self.images + [
                f"{self.root}/{droot}/{ip}" for ip in image_paths
            ]

        self._length = len(self.images)

        if augmentation:
            self.augmentation = albumentations.OneOf([
                albumentations.VerticalFlip(p=0.25),
                albumentations.HorizontalFlip(p=0.25),
                albumentations.RandomRotate90(p=0.25),
            ])
        else:
            self.augmentation = None

        size = None if size is not None and size <= 0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=self.interpolation)

            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,
                                                         width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,
                                                         width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.images[i])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
            image = self.preprocessor(image=image)["image"]

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        if self.fld:
            return image, 0

        example = {
            "image": torch.tensor((image / 127.5 - 1.0).astype(np.float32))
        }

        return example
