import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.logging_configuration import log_handler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

ROOT_PATH = (
    Path(f"../artifacts/ecg/long_tailed_recognition")
    if Path(f"../artifacts").exists()
    else Path(f"/gpfs/projects/bsc70/hpai/storage/long_tailed_recognition")
)


class FeaturesDataset(Dataset):
    FILE_PATH = ROOT_PATH
    PTBXL_RHYTHM_CLASS_LABELS = {
        "SBRAD": 0,
        "SR": 1,
        "AFIB": 2,
        "STACH": 3,
        "AFLT": 4,
        "SARRH": 5,
        "SVTAC": 6,
    }

    def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray
    ):
        super().__init__()
        self.features = torch.Tensor(data).float()
        self.classes = torch.Tensor(labels).long()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self):
        return self.classes.unique(return_counts=True)


if __name__ == "__main__":
    ds = FeaturesDataset("ptbxl_1dcnn")
    log.info(ds.features.shape)
