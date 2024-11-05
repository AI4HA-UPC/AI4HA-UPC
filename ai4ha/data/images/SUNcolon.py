import logging
import os
from typing import Union, Sequence, List, Optional

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Grayscale, RandomHorizontalFlip, RandomRotation
import torchvision.transforms as T

import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

TRAIN_CASES = [
    'case18_2', 'case1_1', 'case1_3', 'case20_1', 'case20_10', 'case20_11', 'case20_12', 'case20_13', 'case20_14',
    'case20_15', 'case20_16', 'case20_17', 'case20_18', 'case20_19', 'case20_2', 'case20_20', 'case20_3', 'case20_4',
    'case20_5', 'case21', 'case22', 'case26_1', 'case26_2', 'case2_1', 'case2_2', 'case2_4', 'case2_7', 'case33_1',
    'case33_2', 'case33_3', 'case35_4', 'case35_5', 'case35_6', 'case35_7', 'case38_1', 'case44_2', 'case44_3',
    'case44_4', 'case45_1', 'case45_2', 'case45_3', 'case46', 'case47_1', 'case49', 'case53', 'case55_1', 'case55_2',
    'case55_3', 'case59_1', 'case59_2', 'case59_3', 'case5_1', 'case5_2', 'case5_3', 'case5_5', 'case61_1', 'case62_1',
    'case62_2', 'case63_1', 'case65', 'case66_1', 'case66_7', 'case6_1', 'case71_3', 'case71_4', 'case71_6', 'case72_2',
    'case73_1', 'case73_2', 'case73_3', 'case73_4', 'case73_5', 'case73_6', 'case73_7', 'case73_8', 'case73_9',
    'case75_2', 'case76_1', 'case76_2', 'case77', 'case7_1', 'case82', 'case83_1', 'case85_1', 'case85_2', 'case85_3',
    'case85_4', 'case85_5', 'case85_6', 'case87_1', 'case87_2', 'case88_1', 'case90_2', 'case98'
]
VAL_CASES = [
    'case4', 'case25_1', 'case28', 'case41', 'case57', 'case58',
    'case69', 'case78_1', 'case78_2', 'case92_1', 'case92_2'
]
TEST_UNSEEN_CASES = [
    'case29_1', 'case36_1', 'case91_7', 'case51_3', 'case51_10', 'case93_2', 'case13_2', 'case68_4', 'case68_7', 'case43',
    'case42', 'case30', 'case50_4', 'case100', 'case14_3', 'case14_6', 'case52', 'case95_1', 'case36_3', 'case51_2',
    'case70', 'case24_6', 'case39_1', 'case81_2', 'case10_1', 'case97_2', 'case32_1', 'case32_4', 'case12_2', 'case68_6',
    'case50_1', 'case27', 'case39_2', 'case54_1', 'case10_2', 'case93_1', 'case84', 'case13', 'case3', 'case8', 'case12',
    'case9', 'case11', 'case10'
]


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
                 folder='TrainDataset',
                 size=256,
                 negatives: bool = True,
                 cases: Sequence[str] = None,
                 augmentation: bool = True,
                 squarify: bool = False,
                 raw: bool = False,
                 local: bool = False,
                 zeroed_mask: bool = False):

        self.include_negatives = negatives
        self.size = size
        self.n_labels = 2
        self.squarify = squarify
        self.zeroed_mask = zeroed_mask

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
        self.cases = []

        # base_path = f'/gpfs/projects/bsc70/hpai/storage/data/atormos/{size}px/sundatabase'
        if not raw:
            base_path = f'/data/atormos/{size}px/sundatabase'
        elif local:
            base_path = f'/home/bejar/ssdstorage/sundatabase'
        else:
            base_path = f'/data/sunraw/sundatabase'
        masks_root = f'SUN-SEG-Annotation/{folder}/GT'
        root_split_path = os.path.join(base_path, masks_root)

        # If cases is not provided, list all cases in directory
        if cases is None:
            cases = [
                d for d in os.listdir(root_split_path)
                if os.path.isdir(os.path.join(base_path, masks_root, d))
                and d[:4] == 'case'
            ]

        # Iterate over all folders in directory
        for case in os.listdir(root_split_path):
            # If folder included in cases
            case_path = os.path.join(root_split_path, case)
            if os.path.isdir(
                    case_path) and case[:4] == 'case' and case in cases:
                actual_case = case.split('_')[0]
                # Iterate over all images in case
                for img in os.listdir(os.path.join(case_path)):
                    img_name, img_ext = os.path.splitext(img)
                    img_path = os.path.join(case_path, img)
                    if os.path.isfile(img_path) and img_ext == '.png':
                        # Try to find the corresponding original image
                        positive_img_path = os.path.join(
                            base_path, 'positive', actual_case,
                            f'{img_name}.jpg')
                        negative_img_path = os.path.join(
                            base_path, 'negative', actual_case,
                            f'{img_name}.jpg')
                        # Add image information to lists if positive case
                        # (safety check, SUN-SEG is only supposed to have masks for positive cases)
                        if os.path.exists(positive_img_path):
                            self.images.append(positive_img_path)
                            self.masks.append(img_path)

        # If include negatives, iterate over the negative cases
        negative_path = os.path.join(base_path, 'negative')
        if self.include_negatives:
            for case in os.listdir(negative_path):
                # If folder included in cases
                case_path = os.path.join(negative_path, case)
                if os.path.isdir(case_path) and case in {
                        x.split('_')[0]
                        for x in cases
                }:
                    # Iterate over all images in case and add their information to lists
                    # None is added to masks to signify a negative mask
                    for img in os.listdir(os.path.join(case_path)):
                        img_name, img_ext = os.path.splitext(img)
                        img_path = os.path.join(case_path, img)
                        if os.path.isfile(img_path) and img_ext == '.jpg':
                            self.images.append(img_path)
                            self.masks.append(None)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        nomask = False
        if mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            nomask = True
        else:
            mask = Image.open(mask_path)
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

        if self.zeroed_mask and nomask:
            onehot = np.repeat(processed["mask"][:, :, np.newaxis], self.n_labels, axis=2).astype(np.float32)
        else:
            onehot = np.eye(self.n_labels)[processed["mask"]].astype(np.float32)

        example["image"] = (processed["image"] / 127.5 - 1.0).astype(
            np.float32)
        try:
            example["segmentation"] = onehot
            s_onehot = ((onehot / np.max(onehot)) * 2) - 1
            example["imageseg"] = np.concatenate((example["image"], s_onehot),
                                                 axis=2)
            example['aesegmentation'] = s_onehot
        except:
            print(
                f"Error {img_path} {example['image'].shape} {mask_path} {onehot.shape}"
            )

        return example


class ExamplesTrain(SunDataset):
    def __init__(self, size=256,
                       augmentation=False, squarify=False, raw=False, local=False, zeroed_mask=False):
        super().__init__(size=size,  augmentation=augmentation,
                         cases=TRAIN_CASES, squarify=squarify, raw=raw, local=local, zeroed_mask=zeroed_mask)


class ExamplesVal(SunDataset):
    def __init__(self, size=256,
                       augmentation=False, squarify=False, raw=False, local=False, zeroed_mask=False):
        super().__init__(size=size,  augmentation=augmentation,
                         cases=VAL_CASES, squarify=squarify, raw=raw, local=local, zeroed_mask=zeroed_mask)


class ExamplesTest(SunDataset):
    def __init__(self, size=256,
                       augmentation=False, squarify=False, raw=False, local=False, zeroed_mask=False):
        super().__init__(size=size,  augmentation=augmentation,
                         cases=TEST_UNSEEN_CASES, squarify=squarify, raw=raw, local=local, zeroed_mask=zeroed_mask)
