import os
import random
from typing import Union

import lightning as pl
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Grayscale, RandomHorizontalFlip, RandomRotation

from ai4ha.data.dataset import GenericSegmentationDataset
from ai4ha.data.enums import TrainingSplitModality

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    ['TR512', 1, 10, 256],
    ['TR512', 2, 10, 256],
    ['TR512', 3, 15, 256],
    ['TR512aug', 1, 10, 256],
    ['TR1024aug', 1, 10, 256],
    ['DDPM256', 500, 0, 256],  
    ['DDPM256', 1000, 0, 256], 
    ['DDPM256SNR', 100, 0, 256], 
    ['DDPM256SNR', 500, 0, 256], 

]


class LDPolypGenDataset(GenericSegmentationDataset):
    def __init__(self, trans='TR512', temp=1, topk=10, size=256,
                 common_transform: Union[Compose, ToTensor] = ToTensor(),
                 transform: Union[Compose, ToTensor] = ToTensor(),
                 mask_transform: Union[Compose, ToTensor] = ToTensor()):
        """_Initialization method_

        Args:
            trans (str, optional): _Transformer model_. Defaults to 'TR512'.
            temp (int, optional): _Temperature sampling_. Defaults to 1.
            topk (int, optional): _Top K sampling_. Defaults to 10.
            size (int, optional): _image size_. Defaults to 256.
        """

        super(LDPolypGenDataset, self).__init__(common_transform, transform, mask_transform)

        if [trans, temp, topk, size] not in DATASETS:
            raise NameError("LDPOLYP: Unknown dataset")

        # base_path = f'/gpfs/projects/bsc70/bsc70642/Data/LDPolypGen/TT{size}/{trans}/T{temp}_K{topk}'
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

        # self.cases = [f'TT{size}-{trans}-T{temp}-k{topk}' for _ in self.images]
        if 'TR' in trans:
            self.cases = [f'TT{size}-{trans}-T{temp}-k{topk}' for _ in self.images]
        elif 'DD' in trans:
            self.cases = [f'DD{size}-{trans}-S{temp}' for _ in self.images]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        case = self.cases[idx]

        image = torch.from_numpy(np.array(Image.open(img_path)).astype(np.uint8)).type(torch.uint8)

        return image


class LDPolypGenDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LD-Polyp-GEN dataset")
        # parser.add_argument('-b', '--batch-size', type=int, help='Batch size.', default=16)
        # parser.add_argument('--data-augmentation', action='store_true',
        #                     help='If present, perform data augmentation on training input.')
        # parser.add_argument('--num-workers', type=int, help='Amount of workers to use when loading data from datasets.',
        #                     default=128)
        return parent_parser

    def __init__(self, batch_size, subdatasets, training_modality=TrainingSplitModality.TEST_ONLY,
                 data_augmentation=False,
                 num_workers=128):
        super(LDPolypGenDataModule, self).__init__()

        self.batch_size = batch_size
        self.subdatasets = DATASETS if subdatasets is None else subdatasets
        self.num_workers = num_workers
        self.data_augmentation = data_augmentation

        split_ratios = {
            TrainingSplitModality.TRAIN_ONLY: [1, 0, 0],
            TrainingSplitModality.TEST_ONLY: [0, 0, 1],
            TrainingSplitModality.TRAIN_VAL: [0.8, 0.2, 0.0],
            TrainingSplitModality.TRAIN_TEST: [0.8, 0.0, 0.2],
            TrainingSplitModality.TRAIN_VAL_TEST: [0.8, 0.1, 0.1]
        }
        self.train = None
        self.val = None
        self.test = None
        self.training_modality = training_modality
        self.split_ratios = split_ratios[self.training_modality]

        self.save_hyperparameters()

        self.setup_done = False

    def setup(self, stage: str) -> None:
        if not self.setup_done:
            datasets = [LDPolypGenDataset(*args) for args in self.subdatasets]
            images = [img for dataset in datasets for img in dataset.images]
            masks = [m for dataset in datasets for m in dataset.masks]
            cases = [id for dataset in datasets for id in dataset.cases]
            instances = list(zip(images, masks, cases))
            random.shuffle(instances)

            train_val_cut_point = int(len(instances) * self.split_ratios[0])
            val_test_cut_point = train_val_cut_point + int(len(instances) * self.split_ratios[1])
            train_instances = instances[:train_val_cut_point]
            val_instances = instances[train_val_cut_point:val_test_cut_point]
            test_instances = instances[val_test_cut_point:]

            if self.training_modality != TrainingSplitModality.TEST_ONLY:
                self.train = LDPolypGenDataset(
                    *self.subdatasets[0],
                    common_transform=Compose([RandomHorizontalFlip(), RandomRotation(180)]
                                             if self.data_augmentation
                                             else []),
                    transform=Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    mask_transform=Compose([Lambda(lambda mask: Grayscale()(mask) if mask.shape[0] == 3 else mask),
                                            Lambda(lambda mask: (mask > 0.5).float())])
                )
                self.train.images = [img for img, _, _ in train_instances]
                self.train.masks  = [msk for _, msk, _ in train_instances]
                self.train.cases  = [did for _, _, did in train_instances]
            if self.training_modality in [TrainingSplitModality.TRAIN_VAL, TrainingSplitModality.TRAIN_VAL_TEST]:
                self.val = LDPolypGenDataset(
                    *self.subdatasets[0],
                    common_transform=Compose([]),
                    transform=Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    mask_transform=Compose([Lambda(lambda mask: Grayscale()(mask) if mask.shape[0] == 3 else mask),
                                            Lambda(lambda mask: (mask > 0.5).float())])
                )
                self.val.images = [img for img, _, _ in val_instances]
                self.val.masks  = [msk for _, msk, _ in val_instances]
                self.val.cases  = [did for _, _, did in val_instances]
            if self.training_modality in [TrainingSplitModality.TEST_ONLY, TrainingSplitModality.TRAIN_TEST,
                                          TrainingSplitModality.TRAIN_VAL_TEST]:
                self.test = LDPolypGenDataset(
                    *self.subdatasets[0],
                    common_transform=Compose([]),
                    transform=Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    mask_transform=Compose([Lambda(lambda mask: Grayscale()(mask) if mask.shape[0] == 3 else mask),
                                            Lambda(lambda mask: (mask > 0.5).float())])
                )
                self.test.images = [img for img, _, _ in test_instances]
                self.test.masks  = [msk for _, msk, _ in test_instances]
                self.test.cases  = [did for _, _, did in test_instances]

            self.setup_done = True

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
