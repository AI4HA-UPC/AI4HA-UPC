from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT_PATH = (
    Path(f"../artifacts/eeg/data")
    if Path(f"../artifacts").exists()
    else Path(f"/gpfs/projects/bsc70/hpai/storage/data/datasets/raw/eeg")
)


class Sizes(NamedTuple):
    small: float
    medium: float
    large: float
    extra: float
    full: float


DATASET_SIZES = Sizes(0.2, 0.4, 0.6, 0.8, 1)


class IsrucCombined(Dataset):
    REAL_FILE_PATH = ROOT_PATH / "isruc/real"

    def __init__(
        self,
        binary_classes: bool = False,
        only_train: bool = False,
        only_test: bool = False,
        only_validation: bool = False,
        channel: int = 1,  # TODO -1 for all channels
        synthetic_size: float = 1,
    ):

        files_real = Path(self.REAL_FILE_PATH).rglob("*.csv")

        for file in files_real:
            data = pd.read_csv(file, header=None, index_col=False)

            # TODO: let's focus on only one channel
            self.data = data.iloc[:, :3000]
            labels = data.iloc[:, -1]

        self.features = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.classes = torch.Tensor(labels).long()
        if binary_classes:
            self.classes[self.classes != 0] = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self):
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self):
        return self.data.shape[-1] - 1

    def dataset_name(self) -> str:
        return self.__class__.__name__


class SleepEDFCombined(Dataset):
    REAL_FILE_PATH = ROOT_PATH / "sleepEDF/real"

    def __init__(
        self,
        binary_classes: bool = False,
        only_train: bool = False,
        only_test: bool = False,
        only_validation: bool = False,
        channel: int = 1,  # TODO -1 for all channels
        synthetic_size: float = 1,
    ):

        files_real = Path(self.REAL_FILE_PATH).rglob("*.csv")

        for file in files_real:
            data = pd.read_csv(file, header=None, index_col=False)

            # TODO: let's focus on only one channel
            self.data = data.iloc[:, :3000]
            labels = data.iloc[:, -1]

        self.features = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.classes = torch.Tensor(labels).long()
        if binary_classes:
            self.classes[self.classes != 0] = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self):
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self):
        return self.data.shape[-1] - 1

    def dataset_name(self) -> str:
        return self.__class__.__name__


def get_dataset(configurations: dict) -> tuple:
    dataset = configurations.get("dataset")
    if dataset == "isruc_combined":
        if configurations["use_synthetic"]:
            None
        else:
            dataset = IsrucCombined()
    if dataset == "sleepedf_combined":
        if configurations["use_synthetic"]:
            None
        else:
            dataset = SleepEDFCombined()
    else:
        raise FileNotFoundError
    return dataset


if __name__ == "__main__":
    # dl = IsrucCombined()
    dl = SleepEDFCombined()
    print(dl.dataset_name())
