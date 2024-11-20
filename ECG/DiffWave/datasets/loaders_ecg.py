import copy
import logging
import sys
from abc import ABC
from pathlib import Path
from typing import NamedTuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.logging_configuration import log_handler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

ROOT_PATH = (
    Path(f"../artifacts/ecg/data")
    if Path(f"../artifacts").exists()
    else Path(f"/gpfs/projects/bsc70/hpai/storage/data/datasets/raw/ecg")
)


class Sizes(NamedTuple):
    small: float
    medium: float
    large: float
    extra: float
    full: float


DATASET_SIZES: Sizes = Sizes(0.2, 0.4, 0.6, 0.8, 1)


class MitEcg:
    BASE_PATH = ROOT_PATH / f"mit-bih-arrhythmia-database-1.0.0"

    def __init__(self):
        # TODO organize by patient
        files = self.BASE_PATH.rglob("*.dat")
        self.data = np.array([])
        for file in files:
            self.data = np.fromfile(file)

    def show_data(self):
        log.info(self.data[:10])
        sys.stdout.flush()


class KagglePTB(Dataset):
    BASE_PATH = ROOT_PATH / "kaggle/ptb-diagnostic"

    def __init__(self, dataset_size: Sizes = None):
        ptb_parts = {}
        aux = []
        files = self.BASE_PATH.rglob("*.csv")
        if not dataset_size:
            dataset_size = DATASET_SIZES.full
        for file in files:
            if file.stem.split("_")[-1] == "normal":
                ptb_parts["normal"] = pd.read_csv(file)
                aux.append(ptb_parts["normal"].T.reset_index(drop=True).T)
            elif file.stem.split("_")[-1] == "abnormal":
                ptb_parts["abnormal"] = pd.read_csv(file)
                aux.append(ptb_parts["abnormal"].T.reset_index(drop=True).T)
            else:
                log.error(f"Unknown file found: {file}")
        self.data = pd.concat(aux).reset_index(drop=True)
        if dataset_size:
            class_weights = 1 - self.data.iloc[:, -1].value_counts() / len(self.data)
            weights = self.data.iloc[:, -1].apply(lambda x: class_weights[x])
            self.data = self.data.sample(frac=dataset_size, weights=weights)
        self.data = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.features = self.data[:, :, :-1]
        self.classes = self.data[:, :, -1].squeeze().long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self) -> int:
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self) -> int:
        return self.data.shape[-1] - 1

    def dataset_name(self) -> str:
        return self.__class__.__name__


class KagglePTBSynthetic(Dataset):
    FILE_PATH = ROOT_PATH / "kaggle/ptb-diagnostic-synthetic"

    def __init__(self, synthetic_dataset_file=None):
        files = self.FILE_PATH.rglob("*.csv")
        aux = []
        if synthetic_dataset_file:
            files = self.FILE_PATH.rglob(synthetic_dataset_file)
        for file in files:
            aux.append(
                pd.read_csv(file, header=None, index_col=False)
                .T.reset_index(drop=True)
                .T
            )
        self.data = pd.concat(aux)
        self.data = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.features = self.data[:, :, :-1]
        self.classes = self.data[:, :, -1].squeeze().long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self) -> int:
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self) -> int:
        return self.data.shape[-1] - 1

    def dataset_name(self) -> str:
        return self.__class__.__name__


class KaggleMIT(Dataset):
    BASE_PATH = ROOT_PATH / "kaggle/mit-bih"
    CLASSES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

    def __init__(
        self,
        binary_classes: bool = False,
        dataset_size: Sizes = None,
        balanced: bool = False,
        only_test: bool = False,
    ):
        if not dataset_size:
            dataset_size = DATASET_SIZES.full
        mit_parts = {}
        files = self.BASE_PATH.rglob("*train.csv")
        aux = []
        if only_test:
            dataset_size = DATASET_SIZES.full
            files = self.BASE_PATH.rglob("*test.csv")
        for file in files:
            last_part = file.stem.split("_")[-1]
            if last_part == "train":
                mit_parts["train"] = pd.read_csv(file, header=None)
                aux.append(mit_parts["train"].T.reset_index(drop=True).T)
            elif last_part == "test":
                mit_parts["test"] = pd.read_csv(file, header=None)
                aux.append(mit_parts["test"].T.reset_index(drop=True).T)
            else:
                log.error(f"Unknown file found: {file}")
        self.data = pd.concat(aux).reset_index(drop=True)
        if dataset_size:
            class_weights = 1 - self.data.iloc[:, -1].value_counts() / len(self.data)
            weights = self.data.iloc[:, -1].apply(lambda x: class_weights[x])
            self.data = self.data.sample(frac=dataset_size, weights=weights)
        if balanced:
            classes = self.data.iloc[:, -1].unique()
            max_obs_class = self.data.iloc[:, -1].value_counts().min()
            aux = []
            for c in classes:
                subset = self.data[self.data.iloc[:, -1] == c]
                aux.append(
                    pd.DataFrame(data=np.random.permutation(subset)[:max_obs_class])
                )
            self.data = pd.concat(aux).reset_index(drop=True)

        self.data = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.features = self.data[:, :, :-1]
        self.classes = self.data[:, :, -1].squeeze().long()
        if binary_classes:
            self.classes[self.classes != 0] = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def get_class_representation(self) -> int:
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self) -> int:
        return self.data.shape[-1] - 1

    def dataset_name(self) -> str:
        return self.__class__.__name__


class KaggleMITSynthetic(Dataset):
    FILE_PATH = ROOT_PATH / "kaggle/mit-bih-synthetic"

    def __init__(self, binary_classes: bool = False, synthetic_dataset_file=None):
        files = self.FILE_PATH.rglob("*.csv")
        aux = []
        if synthetic_dataset_file:
            files = self.FILE_PATH.rglob(synthetic_dataset_file)
        for file in files:
            aux.append(
                pd.read_csv(file, header=None, index_col=False)
                .T.reset_index(drop=True)
                .T
            )
        self.data = pd.concat(aux)
        self.data = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.features = self.data[:, :, :-1]
        self.classes = self.data[:, :, -1].squeeze().long()
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


class KaggleMITAll(Dataset):
    SYNTHETIC_FILE_PATH = ROOT_PATH / "kaggle/mit-bih-synthetic"
    REAL_FILE_PATH = ROOT_PATH / "kaggle/mit-bih"

    def __init__(self, binary_classes: bool = False, only_test: bool = False):
        mit_parts = {}
        aux = []
        if only_test:
            files = self.REAL_FILE_PATH.rglob("*test.csv")
        else:
            files = self.SYNTHETIC_FILE_PATH.rglob("*.csv")
            for file in files:
                aux.append(
                    pd.read_csv(file, header=None, index_col=False)
                    .T.reset_index(drop=True)
                    .T
                )
            files = self.REAL_FILE_PATH.rglob("*train.csv")
        for file in files:
            last_part = file.stem.split("_")[-1]
            if last_part == "train":
                mit_parts["train"] = pd.read_csv(file)
            elif last_part == "test":
                mit_parts["test"] = pd.read_csv(file)
            else:
                log.error(f"Unknown file found: {file}")
        for part in mit_parts.values():
            aux.append(part.T.reset_index(drop=True).T)
        self.data = pd.concat(aux)
        self.data = torch.Tensor(self.data.values).float().unsqueeze(1)
        self.features = self.data[:, :, :-1]
        self.classes = self.data[:, :, -1].squeeze().long()
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


class ECGDataset(ABC, Dataset):
    FILE_PATH = ROOT_PATH
    CHAPMAN_CLASS_LABELS = {
        "SBRAD": 0,
        "SR": 1,
        "AFIB": 2,
        "STACH": 3,
        "AFLT": 4,
        "SARRH": 5,
        "SVTAC": 6,
    }
    METADATA_COLUMNS = None

    def __init__(
        self, split: str, sample_frequency: int, binary_classes: bool = False
    ) -> None:
        self.classes = None
        self.features = None
        self.mask = None
        self.split = split
        self.means = None
        self.stds = None
        self.sample_frequency = sample_frequency
        self.original_classes = None
        super().__init__()
        self.binary_classes = binary_classes

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.classes[index]
        return x, y

    def _final_reshape(self, data):
        data = data.squeeze()
        self.features = data[:, :, :-1]
        self.classes = data[:, :, -1].squeeze().long()[:, 0]

    def get_class_representation(self):
        return self.classes.unique(return_counts=True)

    def get_steps_per_sequence(self):
        return self.features.shape[-1]

    def get_height_per_sequence(self):
        if self.features.dim() > 3:
            height = self.features.shape[-2]
        else:
            height = 1
        return height

    def get_depth_per_sequence(self):
        if self.features.dim() > 3:
            depth = self.features.shape[-3]
        else:
            depth = self.features.shape[-2]
        return depth

    def dataset_name(self) -> str:
        return self.__class__.__name__

    def remove_classes(self, labels):
        for label in labels:
            log.critical(
                f"Removing class {label} with representation lower than 100 samples"
            )
            self.mask = self.classes != label
            self.classes = self.classes[self.mask]
            self.features = self.features[self.mask]

    def standardize_data(self, means: List[float], stds: List[float]):
        self.features = (self.features - means) / stds

    def compute_standardize_parameters(self):
        self.means = self.features.mean((0, -1), keepdims=True)
        self.stds = self.features.std((0, -1), keepdims=True)

    def _save_standardize_parameters(self):
        standardize_parameters_path = self.FILE_PATH / "standardize"
        standardize_parameters_path.mkdir(exist_ok=True)
        with open(standardize_parameters_path / "means.npy", mode="wb") as fp:
            np.save(fp, self.means)
        with open(standardize_parameters_path / "stds.npy", mode="wb") as fp:
            np.save(fp, self.stds)

    def _load_standardize_parameters(self):
        standardize_parameters_path = self.FILE_PATH / "standardize"
        with open(standardize_parameters_path / "means.npy", mode="rb") as fp:
            self.means = torch.Tensor(np.load(fp))
        with open(standardize_parameters_path / "stds.npy", mode="rb") as fp:
            self.stds = torch.Tensor(np.load(fp))

    def copy_dataset(self, dataset_size: Sizes = 1):
        new_dataset = copy.deepcopy(self)
        if dataset_size < 1:
            class_weights = 1 - new_dataset.get_class_representation()[1] / len(
                new_dataset.classes
            )
            aux = pd.DataFrame(new_dataset.classes)
            weights = aux.apply(lambda x: class_weights[x]).values.squeeze()
            sample_idx = aux.sample(frac=dataset_size, weights=weights).index
            new_dataset.features = new_dataset.features[sample_idx]
            new_dataset.classes = new_dataset.classes[sample_idx]
        return new_dataset

    def binarize_dataset(self, positive_class: int):
        d = {}
        for i in self.classes.unique().numpy():
            if i != positive_class:
                d[i] = 1
            else:
                d[i] = 0
        self.classes.apply_(d.get)

    def specific_class(self, class_label):
        mask = self.original_classes == class_label
        self.features = self.features[mask]
        self.classes = self.classes[mask]


class PTBXLRhythm(ECGDataset):
    FILE_PATH = ROOT_PATH / "processed/ptbxl"
    PTBXL_RHYTHM_CLASS_LABELS = {
        "SBRAD": 0,
        "SR": 1,
        "AFIB": 2,
        "STACH": 3,
        "AFLT": 4,
        "SARRH": 5,
        "SVTAC": 6,
    }
    METADATA_COLUMNS = [
        "ecg_id",
        "age",
        "sex",
        "height",
        "weight",
        "nurse",
        "site",
        "device",
        "recording_date",
        "report",
        "scp_codes",
        "heart_axis",
        "infarction_stadium1",
        "infarction_stadium2",
        "validated_by",
        "second_opinion",
        "initial_autogenerated_report",
        "validated_by_human",
        "baseline_drift",
        "static_noise",
        "burst_noise",
        "electrodes_problems",
        "extra_beats",
        "pacemaker",
        "strat_fold",
        "filename_lr",
        "filename_hr",
        "label",
        "lead",
    ]
    lead_column_name = {"0": "lead"}

    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize=False,
    ):
        super().__init__(split, sample_frequency)
        self.split = split
        files = self.FILE_PATH.rglob(
            f"ptb_xl_dataset_rhythm_original_frequency_{sample_frequency}*_all_leads_*{split}.csv"
        )

        for file in files:
            data = pd.read_csv(
                file,
                header=0,
            )
            data.rename({"0": "lead"}, axis=1, inplace=True)
            metadata_columns = self.METADATA_COLUMNS.copy()
            data = data.set_index("patient_id")
            data = data.drop(data[metadata_columns], axis=1)
            na_idx = data.loc[pd.isna(data).any(1), :].index
            data = data.drop(na_idx, axis=0)
            data = data.reset_index(drop=True)

            data = data.values.reshape(int(data.shape[0] / 12), 1, 12, data.shape[-1])
            data = torch.Tensor(data).float()

            self._final_reshape(data)

            if standardize:
                self.compute_standardize_parameters()
                self._save_standardize_parameters()
                self.features = (self.features - self.means) / self.stds

            self.original_classes = self.classes.clone()
            if binary_classes:
                self.binarize_dataset(positive_class=1)


class PTBXLSyntheticRhythm(ECGDataset):
    SYNTHETIC_FILE_PATH = ROOT_PATH / "ptbxl/synthetic"
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
        split: str,
        binary_classes: bool = False,
        sample_frequency=100,
        generative_model: str = "diffwave",
        standardize: bool = False,
    ):
        super().__init__(split, sample_frequency)
        self.split = split

        file = (
            self.SYNTHETIC_FILE_PATH / f"ptbxl_combined_{generative_model}_{split}.npy"
        )
        try:
            with open(file, mode="rb") as fp:
                data = np.load(fp)
        except FileNotFoundError as e:
            log.error(
                f"File not found: {file}. Please preprocess synthetic data using split_synthetic_data.py methods"
            )
            exit(1)
        data = torch.Tensor(data).float()
        self._final_reshape(data)

        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds
        if binary_classes:
            self.binarize_dataset(positive_class=1)


class CHAPMANSynthetic(ECGDataset):
    SYNTHETIC_FILE_PATH = ROOT_PATH / "chapman/synthetic"
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
        split: str,
        binary_classes: bool = False,
        sample_frequency=100,
        generative_model: str = "diffwave",
        standardize: bool = False,
    ):
        super().__init__(split, sample_frequency)
        self.split = split

        file = Path(
            self.SYNTHETIC_FILE_PATH
            / f"chapman_combined_{generative_model}_{split}.npy"
        )
        try:
            with open(file, mode="rb") as fp:
                data = np.load(fp)
        except FileNotFoundError as e:
            log.error(
                f"File not found: {file}. Please preprocess synthetic data using split_synthetic_data.py methods"
            )
            exit(1)
        data = torch.Tensor(data).float()
        # indices = torch.concat([torch.where(data[:, :, -1] == idx)[0] for idx in [2, 5, 6]], dim=0).ravel()
        # indices = torch.unique(indices)
        # data = data[indices]
        self._final_reshape(data)
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds
        if binary_classes:
            self.binarize_dataset(positive_class=1)


class PTBXLRealAndSyntheticRhythm(ECGDataset):
    def __init__(
        self,
        split: str,
        binary_classes: bool = False,
        sample_frequency=100,
        generative_model: str = "diffwave",
        standardize: bool = False,
        two_sample_test: bool = False,
    ):
        super().__init__(split, sample_frequency)
        self.split = split
        real_data = PTBXLRhythm(
            binary_classes=binary_classes,
            sample_frequency=sample_frequency,
            split=split,
            standardize=False,
        )
        synthetic_data = PTBXLSyntheticRhythm(
            split=split,
            binary_classes=binary_classes,
            sample_frequency=sample_frequency,
            standardize=False,
            generative_model=generative_model,
        )
        self.original_classes = torch.concat(
            [real_data.classes, synthetic_data.classes]
        )
        if two_sample_test:
            real_data.classes = torch.zero_(real_data.classes)
            synthetic_data.classes = torch.zero_(synthetic_data.classes) + 1

        self.features = torch.concat([real_data.features, synthetic_data.features])
        self.classes = torch.concat([real_data.classes, synthetic_data.classes])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


class CHAPMANRealAndSynthetic(ECGDataset):
    def __init__(
        self,
        split: str,
        binary_classes: bool = False,
        sample_frequency=100,
        generative_model: str = "diffwave",
        standardize: bool = False,
        two_sample_test: bool = False,
    ):
        super().__init__(split, sample_frequency)
        self.split = split
        real_data = CHAPMAN(
            split=split,
            binary_classes=binary_classes,
            sample_frequency=sample_frequency,
            standardize=False,
        )
        synthetic_data = CHAPMANSynthetic(
            split=split,
            binary_classes=binary_classes,
            sample_frequency=sample_frequency,
            standardize=False,
            generative_model=generative_model,
        )
        self.original_classes = torch.concat(
            [real_data.classes, synthetic_data.classes]
        )
        if two_sample_test:
            real_data.classes = torch.zero_(real_data.classes)
            synthetic_data.classes = torch.zero_(synthetic_data.classes) + 1

        self.features = torch.concat([real_data.features, synthetic_data.features])
        self.classes = torch.concat([real_data.classes, synthetic_data.classes])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


class CHAPMAN(ECGDataset):
    FILE_PATH = ROOT_PATH / "processed/chapman"
    CHAPMAN_CLASS_LABELS = {
        "SBRAD": 0,
        "SR": 1,
        "AFIB": 2,
        "STACH": 3,
        "AFLT": 4,
        "SARRH": 5,
        "SVTAC": 6,
    }
    METADATA_COLUMNS = ["age", "sex", "lead", "id"]
    LABEL_COLUMN_NAME = "diagnosis"

    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
    ):
        super().__init__(split, sample_frequency)
        files = Path(self.FILE_PATH).rglob(
            f"chapman_data_freq_{sample_frequency}_all_leads_ptbxl_labels_*{split}.csv"
        )
        for file in files:
            data = pd.read_csv(
                file,
                header=0,
            )
            metadata_columns = self.METADATA_COLUMNS.copy()
            data = data.drop(data[["age"]], axis=1)
            data = data.set_index("id")
            na_idx = data.loc[pd.isna(data).any(1), :].index
            data = data.drop(na_idx, axis=0)
            data = data.reset_index()
            metadata_columns.remove("age")
            data = data.drop(data[metadata_columns], axis=1)

            data = data.values.reshape(int(data.shape[0] / 12), 1, 12, data.shape[-1])
            data = torch.Tensor(data).float()
            self._final_reshape(data)
            if standardize:
                self.compute_standardize_parameters()
                self._save_standardize_parameters()
                self.features = (self.features - self.means) / self.stds
        self.original_classes = self.classes.clone()

        if binary_classes:
            self.binarize_dataset(positive_class=1)


class PTBXLAndCHAPMANReal(ECGDataset):
    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
    ):
        super().__init__(split, sample_frequency)
        data = [
            PTBXLRhythm(split=split, standardize=False, binary_classes=binary_classes),
            CHAPMAN(split=split, standardize=False, binary_classes=binary_classes),
        ]
        self.features = torch.concat([sd.features for sd in data])
        self.classes = torch.concat([sd.classes for sd in data])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


class PTBXLAndCHAPMANSynthetic(ECGDataset):
    PTBXL_SYNTHETIC_FILE_PATH = Path("/home/jam/PycharmProjects/ai4ha-eeg-ecg/artifacts/ecg/data/ptbxl/synthetic")
    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
        generative_model: str = "diffwave",
    ):
        super().__init__(split, sample_frequency)
        self.split = split

        file = (
                self.PTBXL_SYNTHETIC_FILE_PATH / f"ptbxl_chapman_combined_{generative_model}_{split}.npy"
        )
        try:
            with open(file, mode="rb") as fp:
                data = np.load(fp)
        except FileNotFoundError as e:
            log.error(
                f"File not found: {file}. Please preprocess synthetic data using split_synthetic_data.py methods"
            )
            exit(1)
        data = torch.Tensor(data).float()
        self._final_reshape(data)

        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds
        if binary_classes:
            self.binarize_dataset(positive_class=1)


class PTBXLAndCHAPMANRealAndSynthetic(ECGDataset):
    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
        generative_model: str = "diffwave",
    ):
        super().__init__(split, sample_frequency)
        data = [
            PTBXLAndCHAPMANReal(split=split, standardize=False, binary_classes=binary_classes),
            PTBXLAndCHAPMANSynthetic(split=split, standardize=False, binary_classes=binary_classes,
                                     generative_model=generative_model)
        ]
        self.features = torch.concat([sd.features for sd in data])
        self.classes = torch.concat([sd.classes for sd in data])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


class PTBXLRealCHAPMANSynthetic(ECGDataset):
    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
        generative_model: str = "diffwave",
    ):
        super().__init__(split, sample_frequency)
        data = [
            PTBXLRhythm(split=split, standardize=False, binary_classes=binary_classes),
            CHAPMANSynthetic(
                split=split,
                standardize=False,
                binary_classes=binary_classes,
                generative_model=generative_model,
            ),
        ]
        self.features = torch.concat([sd.features for sd in data])
        self.classes = torch.concat([sd.classes for sd in data])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


class PTBXLSyntheticCHAPMANReal(ECGDataset):
    def __init__(
        self,
        binary_classes: bool = False,
        sample_frequency: int = 100,
        split: str = "train",
        standardize: bool = False,
        generative_model: str = "diffwave",
    ):
        super().__init__(split, sample_frequency)
        data = [
            PTBXLSyntheticRhythm(
                split=split,
                standardize=False,
                binary_classes=binary_classes,
                generative_model=generative_model,
            ),
            CHAPMAN(split=split, standardize=False, binary_classes=binary_classes),
        ]
        self.features = torch.concat([sd.features for sd in data])
        self.classes = torch.concat([sd.classes for sd in data])
        if standardize:
            self.compute_standardize_parameters()
            self.features = (self.features - self.means) / self.stds


def get_dataset(configurations: dict, experiment_setting: str) -> tuple:
    dataset = configurations.get("dataset")
    if dataset == "ptb-diagnostic":
        if configurations["use_synthetic"]:
            dataset = KagglePTBSynthetic()
        else:
            dataset = (
                KagglePTB(dataset_size=configurations["dataset_size"])
                if configurations.get("dataset_size")
                else KagglePTB()
            )
    elif dataset == "ptbxl":
        if experiment_setting == "TrSTeS":
            dataset_train = PTBXLSyntheticRhythm(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLSyntheticRhythm(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLSyntheticRhythm(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrSTeR":
            dataset_train = PTBXLSyntheticRhythm(
                split="train",
                generative_model=configurations["generative_model"],
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLSyntheticRhythm(
                split="validation",
                generative_model=configurations["generative_model"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLRhythm(split="test")
        elif experiment_setting == "TrRTeS":
            dataset_train = PTBXLRhythm(
                split="train",
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLRhythm(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLSyntheticRhythm(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrRSTeR":
            dataset_train = PTBXLRealAndSyntheticRhythm(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLRealAndSyntheticRhythm(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLRhythm(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRSTeRS":
            dataset_train = PTBXLRealAndSyntheticRhythm(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
            dataset_validation = PTBXLRealAndSyntheticRhythm(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
            dataset_test = PTBXLRealAndSyntheticRhythm(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
        else:
            dataset_train = PTBXLRhythm(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLRhythm(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLRhythm(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        dataset = (dataset_train, dataset_validation, dataset_test)
    elif dataset == "chapman":
        if experiment_setting == "TrSTeS":
            dataset_train = CHAPMANSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = CHAPMANSynthetic(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMANSynthetic(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrSTeR":
            dataset_train = CHAPMANSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = CHAPMANSynthetic(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMAN(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRTeS":
            dataset_train = CHAPMAN(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = CHAPMAN(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMANSynthetic(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrRSTeR":
            dataset_train = CHAPMANRealAndSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = CHAPMANRealAndSynthetic(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
                generative_model=configurations.get("generative_model"),
            )
            dataset_test = CHAPMAN(
                split="test",
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrRSTeRS":
            dataset_train = CHAPMANRealAndSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
            dataset_validation = CHAPMANRealAndSynthetic(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
            dataset_test = CHAPMANRealAndSynthetic(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
                two_sample_test=configurations.get("two_sample_test"),
            )
        else:
            dataset_train = CHAPMAN(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = CHAPMAN(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMAN(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        dataset = (dataset_train, dataset_validation, dataset_test)
    elif dataset == "ptbxl_chapman":
        if experiment_setting == "TrSTeS":
            dataset_train = PTBXLAndCHAPMANSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANSynthetic(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLAndCHAPMANSynthetic(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrSTeR":
            dataset_train = PTBXLAndCHAPMANSynthetic(
                split="train",
                generative_model=configurations.get("generative_model"),
                standardize=configurations["standardize"],
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANSynthetic(
                split="validation",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLAndCHAPMANReal(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRTeS":
            dataset_train = PTBXLAndCHAPMANReal(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANReal(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLAndCHAPMANSynthetic(
                split="test",
                generative_model=configurations.get("generative_model"),
                binary_classes=configurations.get("binary_classification"),
            )
        elif experiment_setting == "TrRSTeR":
            dataset_train = PTBXLAndCHAPMANRealAndSynthetic(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANRealAndSynthetic(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLAndCHAPMANReal(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRTrSTeRNUL":
            dataset_train = PTBXLRealCHAPMANSynthetic(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLRealCHAPMANSynthetic(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLRhythm(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrSTrRNULTeR":
            dataset_train = PTBXLSyntheticCHAPMANReal(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLSyntheticCHAPMANReal(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMAN(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRTrRTeRNUL":
            dataset_train = PTBXLAndCHAPMANReal(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANReal(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLRhythm(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        elif experiment_setting == "TrRTrRNULTeR":
            dataset_train = PTBXLAndCHAPMANReal(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANReal(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = CHAPMAN(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        else:
            dataset_train = PTBXLAndCHAPMANReal(
                split="train",
                standardize=configurations.get("standardize"),
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_validation = PTBXLAndCHAPMANReal(
                split="validation",
                binary_classes=configurations.get("binary_classification"),
            )
            dataset_test = PTBXLAndCHAPMANReal(
                split="test", binary_classes=configurations.get("binary_classification")
            )
        dataset = (dataset_train, dataset_validation, dataset_test)

    else:
        # "mit-bih"
        if configurations["binary_classification"]:
            dataset_train = KaggleMIT(binary_classes=True)
            dataset_test = KaggleMIT(binary_classes=True, only_test=True)
            dataset = (dataset_train, dataset_test)
        elif configurations["use_synthetic"]:
            dataset = KaggleMITSynthetic()
        elif configurations["use_real_and_synthetic"]:
            dataset_train = KaggleMITAll()
            dataset_test = KaggleMITAll(only_test=True)
            dataset = dataset_train, dataset_test
        else:
            dataset_train = (
                KaggleMIT(dataset_size=configurations["dataset_size"])
                if configurations.get("dataset_size")
                else KaggleMIT()
            )
            dataset_test = (
                KaggleMIT(dataset_size=configurations["dataset_size"], only_test=True)
                if configurations.get("dataset_size")
                else KaggleMIT(only_test=True)
            )
            dataset = (dataset_train, dataset_test)
    return dataset


if __name__ == "__main__":
    # s = "train"
    # d = PTBXLSyntheticCHAPMANReal(split=s, standardize=True)
    # log.info(
    #     f"PTBXL synthetic with CHAPMAN real in one: {d.get_class_representation()}"
    # )
    # d = PTBXLRealCHAPMANSynthetic(split=s, standardize=True)
    # log.info(
    #     f"PTBXL real with CHAPMAN synthetic in one: {d.get_class_representation()}"
    # )
    #
    # d = PTBXLSyntheticRhythm(split=s, standardize=True)
    # j = d.copy_dataset(dataset_size=0.2)
    # log.info(
    #     f"Copied dataset with class representation: {j.get_class_representation()}"
    # )
    # d = PTBXLAndCHAPMANReal(split=s, standardize=True)
    # log.info(f"PTBXL and CHAPMAN in one: {d.get_class_representation()}")
    # d = PTBXLRhythm(sample_frequency=100, split=s, standardize=True)
    # log.info(d.get_class_representation())
    # log.info(d.features.std((0, -1)))
    # log.info(len(d))
    # d = CHAPMAN(split=s, sample_frequency=100, standardize=True)
    # log.info(d.get_class_representation())
    # log.info(len(d))
    # log.info(d.features.std((0, -1)))
    # d = CHAPMANRealAndSynthetic(split="validation")
    # log.info(
    #     f"Real and synthetic from CHAPMAN class representation {d.get_class_representation()}"
    # )

    d = PTBXLRealAndSyntheticRhythm(split="validation")
    log.info(
        f"Real and synthetic from PTBXL class representation {d.get_class_representation()}"
    )

    for s in ["train", "test", "validation"]:
        for g in ["diffwave", "time_diffusion", "time_vqvae"]:
            d = PTBXLSyntheticRhythm(split=s, generative_model=g, standardize=True)
            log.info(f"Synthetic data shape: {d.features.shape}")
            d = CHAPMANSynthetic(split=s, generative_model=g, standardize=True)
            log.info(f"Synthetic data shape: {d.features.shape}")

    # CHAPMANSynthetic(sample_frequency=100, dimensions=2)
    # Chapman(sample_frequency=100, dimensions=2)
