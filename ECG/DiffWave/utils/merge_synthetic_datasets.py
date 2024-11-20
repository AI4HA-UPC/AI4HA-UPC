import logging

import numpy as np
from pathlib import Path
from utils.logging_configuration import log_handler
import torch

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

ROOT_PATH = (
    Path(f"../artifacts/ecg/data")
    if Path(f"../artifacts").exists()
    else Path(f"/gpfs/projects/bsc70/hpai/storage/data/datasets/raw/ecg")
)
GENERATIVE_MODELS = ["diffwave", "time_diffusion", "time_vqvae"]
SPLITS = ["train", "validation", "test"]


def merge_synthetic_datasets(datasets: list):
    for dataset_name in datasets:
        dataset_path = ROOT_PATH / dataset_name / "synthetic"
        for s in SPLITS:
            data = []
            for gm in GENERATIVE_MODELS:
                file_path = dataset_path / f"{dataset_name}_combined_{gm}_{s}.npy"
                with open(file_path, mode="rb") as fp:
                    data.append(torch.Tensor(np.load(fp)))
            save_path = dataset_path / f"{dataset_name}_combined_all_{s}.npy"
            with open(save_path, mode="wb") as fp:
                np.save(fp, torch.concat(data))


if __name__ == "__main__":
    original_datasets = ["ptbxl", "chapman"]
    merge_synthetic_datasets(original_datasets)
