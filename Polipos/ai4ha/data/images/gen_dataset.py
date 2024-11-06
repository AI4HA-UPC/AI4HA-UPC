import logging
import os
import random

import lightning as pl
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy

# from src.utils import TrainingSplitModality

log = logging.getLogger(__name__)


DATASETS = [
    ['TR512', 1, 10, 256],
    ['TR512', 2, 10, 256],
    ['TR512', 3, 15, 256],
    ['TR512aug', 1, 10, 256],
    ['TR1024aug', 1, 10, 256],
    ['TR1024aug', 1, 10, 256], 
    ['DDPM256', 500, 0, 256], 
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
            base_path = f'/gpfs/projects/bsc70/bsc70642/Data/LDPolypGen/TT{size}/{trans}/T{temp}_K{topk}'
        elif 'DD' in trans:
            base_path = f'/gpfs/projects/bsc70/bsc70642/Data/LDPolypGen/DD{size}/{trans}/S{temp}'

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
        print( img_path, mask_path )

        image = Image.open(img_path)
        if self.normalize:
            image = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)
        else:
            image = Compose([ToTensor()])(image)

        mask = numpy.asarray(Image.open(mask_path), dtype=numpy.int64)
        mask = torch.from_numpy(mask[:, :, 1]//255).float()
        mask = mask.unsqueeze(0)

        return image, mask, self.data_id[idx]


# class LDPolypGenDatasetALL(LDPolypGenDataset):
#     def __init__(self, normalize=False):
#         datasets = [LDPolypGenDataset(*args, normalize) for args in DATASETS]

#         self.images = [img for dataset in datasets for img in dataset.images]
#         self.masks = [m for dataset in datasets for m in dataset.masks]
#         self.data_id = [id for dataset in datasets for id in dataset.data_id]

#         self.normalize = normalize


# class LDPolypGenDataModule(pl.LightningDataModule):
#     @staticmethod
#     def add_data_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("LDPolypGenData")
#         parser.add_argument('-b', '--batch-size', type=int, help='Batch size.', default=16)
#         parser.add_argument('-s', '--image-size', type=int, help='Image size to use.', default=256)
#         parser.add_argument('--normalize', action='store_true', help='If present, normalize images in pre-processing.')
#         parser.add_argument('--num-workers', type=int, help='Amount of workers to use when loading data from datasets.',
#                             default=128)
#         parser.add_argument('--data-augmentation', action='store_true',
#                             help='If present, perform data augmentation on training input.')
#         return parent_parser

#     def __init__(self, image_size, batch_size, training_modality=TrainingSplitModality.TEST_ONLY,
#                  data_augmentation=False, normalize=False,
#                  num_workers=128):
#         super(LDPolypGenDataModule, self).__init__()

#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.data_augmentation = data_augmentation
#         self.normalize = normalize

#         split_ratios = {
#             TrainingSplitModality.TRAIN_ONLY: [1, 0, 0],
#             TrainingSplitModality.TEST_ONLY: [0, 0, 1],
#             TrainingSplitModality.TRAIN_VAL: [0.8, 0.2, 0.0],
#             TrainingSplitModality.TRAIN_TEST: [0.8, 0.0, 0.2],
#             TrainingSplitModality.TRAIN_VAL_TEST: [0.8, 0.1, 0.1]
#         }
#         self.train = None
#         self.val = None
#         self.test = None
#         self.training_modality = training_modality
#         self.split_ratios = split_ratios[self.training_modality]

#         self.save_hyperparameters()

#         self.setup_done = False

#     def setup(self, stage: str) -> None:
#         if not self.setup_done:
#             d = LDPolypGenDatasetALL(self.normalize)
#             instances = list(zip(d.images, d.masks, d.data_id))
#             random.shuffle(instances)

#             train_val_cut_point = int(len(instances) * self.split_ratios[0])
#             val_test_cut_point = train_val_cut_point + int(len(instances) * self.split_ratios[1])
#             train_instances = instances[:train_val_cut_point]
#             val_instances = instances[train_val_cut_point:val_test_cut_point]
#             test_instances = instances[val_test_cut_point:]

#             if self.training_modality != TrainingSplitModality.TEST_ONLY:
#                 self.train = LDPolypGenDatasetALL(self.normalize)
#                 self.train.images  = [img for img, _, _ in train_instances]
#                 self.train.masks   = [msk for _, msk, _ in train_instances]
#                 self.train.data_id = [did for _, _, did in train_instances]
#             if self.training_modality in [TrainingSplitModality.TRAIN_VAL, TrainingSplitModality.TRAIN_VAL_TEST]:
#                 self.val = LDPolypGenDatasetALL(self.normalize)
#                 self.val.images  = [img for img, _, _ in val_instances]
#                 self.val.masks   = [msk for _, msk, _ in val_instances]
#                 self.val.data_id = [did for _, _, did in val_instances]
#             if self.training_modality in [TrainingSplitModality.TEST_ONLY, TrainingSplitModality.TRAIN_TEST,
#                                           TrainingSplitModality.TRAIN_VAL_TEST]:
#                 self.test = LDPolypGenDatasetALL(self.normalize)
#                 self.test.images  = [img for img, _, _ in test_instances]
#                 self.test.masks   = [msk for _, msk, _ in test_instances]
#                 self.test.data_id = [did for _, _, did in test_instances]

#             self.setup_done = True

#     def train_dataloader(self):
#         return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

#     def val_dataloader(self):
#         return DataLoader(dataset=self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#     def test_dataloader(self):
#         return DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
