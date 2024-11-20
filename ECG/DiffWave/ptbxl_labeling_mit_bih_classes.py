import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.loaders_ecg import get_dataset, KaggleMIT
from utils.dl_utils import load_model, test_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics,
)
from datasets.loaders_ecg import PTBXLRhythm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("configurations/ecg_hparams_test.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)


if __name__ == "__main__":
    dataset = PTBXLRhythm()
    models_path = ARTIFACTS_PATH / "models"
    experiment_name = "train_real_test_real_MIT"
    hparams = EXPERIMENTS["experiments"][experiment_name]
    model_name = hparams["model"]
    dataset_size = len(dataset)
    t_steps_per_sequence = dataset.get_steps_per_sequence()
    model = load_model(
        model_name,
        "ResnetSignalClassificationEEG",
        models_path,
        hparams,
        1,
        1,
        t_steps_per_sequence,
    )
    log.info(f"Loaded model: \n {model}")
    sys.stdout.flush()
    test_loader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=True)
    predictions_targets = test_model(model, test_loader)
    sys.stdout.flush()
    y_var = predictions_targets[0]
    y = predictions_targets[1]
    class_name = KaggleMIT.CLASSES
    for i in range(5):
        print(f"Class {i}-{class_name[i]}")
        counts = np.unique(y[y_var == i], return_counts=True)
        top_classes = np.argpartition(counts[1], -3)[-3:]
        total_representation = np.sum(counts[1])
        percentage_representation = counts[1][top_classes] / total_representation
        most_predicted_class = np.argmax(counts[1])
        most_predicted_class_true_representation = len(
            y[y == most_predicted_class]
        ) / len(y)

        print(f"Counts for predicted class {i}: {counts}")
        print(f"Top classes: {top_classes}")
        print(f"Predicted class representation: {percentage_representation}")
        print(
            f"True class representation: {most_predicted_class_true_representation} \n \n"
        )
