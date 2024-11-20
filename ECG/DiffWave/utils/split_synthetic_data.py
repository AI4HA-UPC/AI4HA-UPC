import argparse
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from einops import rearrange, repeat, pack

from utils.logging_configuration import log_handler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

ROOT_PATH = (
    Path(f"../artifacts/ecg/data")
    if Path(f"../artifacts").exists()
    else Path(f"/gpfs/projects/bsc70/hpai/storage/data/datasets/raw/ecg")
)


def synthetic_data_reshape(synthetic_data: pd.DataFrame, labels: npt.ArrayLike):
    n = len(labels)
    labels = repeat(labels, "l -> (l l2)", l2=12)
    labels = rearrange(labels, "(b c h w) -> b c h w", b=n, c=12, h=1)
    synthetic_data = rearrange(
        synthetic_data.values[:, :-1], "b (c h w) -> b c h w", c=12, h=1, w=1000
    )
    synthetic_data = pack([synthetic_data, labels], "b h w *")[0]
    synthetic_data = synthetic_data.squeeze()
    return synthetic_data


def read_bejar_npz_files(file_path: Path, output_name: str):
    data = np.load(file_path)
    samples = data.get("samples")[:, :, :-24]
    classes = data.get("classes")
    x = [rearrange(samples, "b h w -> b (h w)", h=12), rearrange(classes, "c -> c 1")]
    concat_x = pack(x, "b *")[0]
    df = pd.DataFrame(concat_x)
    df.to_csv(f"{output_name}.csv", index=False)


def synthetic_data_splits(data_path: Path, percentage: float = 0.2):
    data = pd.read_csv(
        data_path,
        header=0,
    )
    n = len(data)
    targets = data.values[:, -1]

    synthetic_splits = Path("utils/synthetic_splits")
    synthetic_splits.mkdir(exist_ok=True)

    labels, label_counts = np.unique(targets, return_counts=True)
    val_test_indices = []
    for l, lc in zip(labels, label_counts):
        val_test_indices.append(
            data[data.iloc[:, -1] == l].sample(n=int(lc * percentage)).index
        )
    val_test_indices = np.concatenate(val_test_indices)
    test_indices = set(
        pd.DataFrame(val_test_indices).sample(frac=0.5).values.squeeze().tolist()
    )
    validation_indices = set(val_test_indices).difference(set(test_indices))
    train_indices = set(data.index).difference(set(val_test_indices))
    assert train_indices.intersection(validation_indices) == set()
    assert train_indices.intersection(test_indices) == set()
    assert validation_indices.intersection(test_indices) == set()
    assert (len(train_indices) + len(validation_indices) + len(test_indices)) == n

    pd.DataFrame(train_indices).to_csv(
        synthetic_splits / "train_indices.csv", header=False, index=False
    )
    pd.DataFrame(validation_indices).to_csv(
        synthetic_splits / "validation_indices.csv", header=False, index=False
    )
    pd.DataFrame(test_indices).to_csv(
        synthetic_splits / "test_indices.csv", header=False, index=False
    )

    data = synthetic_data_reshape(data, targets)
    train_data = data[list(train_indices)]
    validation_data = data[list(validation_indices)]
    test_data = data[list(test_indices)]

    with open(str(file_path.parent / file_path.stem) + "_train.npy", mode="wb") as fp:
        np.save(fp, train_data)
        log.info(f"Created file {str(file_path.parent / file_path.stem) + '_train.npy'}")
    with open(
            str(file_path.parent / file_path.stem) + "_validation.npy", mode="wb"
    ) as fp:
        np.save(fp, validation_data)
        log.info(f"Created file {str(file_path.parent / file_path.stem) + '_validation.npy'}")
    with open(str(file_path.parent / file_path.stem) + "_test.npy", mode="wb") as fp:
        np.save(fp, test_data)
        log.info(f"Created file {str(file_path.parent / file_path.stem) + '_test.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train, val and test splits from a synthetic generated dataset"
    )
    parser.add_argument(
        "--file_path",
        type=Path,
        default="ptbxl/synthetic/ptbxl_chapman_combined_time_diffusion.csv",
        help="Path to the synthetic data.",
    )
    args = parser.parse_args()
    file_path = ROOT_PATH / args.file_path
    log.info(f"Processing file: {file_path}")
    synthetic_data_splits(file_path)
