import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.loaders_ecg import get_dataset
from utils.dl_utils import load_model, test_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics, compute_roc_auc,
)

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
    log.info("Start test models experiments")
    models_path = ARTIFACTS_PATH / "models"
    results_path = ARTIFACTS_PATH / "results"
    plots_path = ARTIFACTS_PATH / "plots"
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        train_dataset, validation_dataset, test_dataset = get_dataset(hparams, "TrRTeR")
        if hparams["standardize"]:
            validation_dataset.standardize_data(
                train_dataset.means, train_dataset.stds
            )
            test_dataset.standardize_data(
                train_dataset.means, train_dataset.stds
            )

        model_name = hparams["model"]
        dataset_size = len(train_dataset)
        t_steps_per_sequence = train_dataset.get_steps_per_sequence()
        h_steps_per_sequence = train_dataset.get_height_per_sequence()
        d_steps_per_sequence = train_dataset.get_depth_per_sequence()
        num_classes = len(train_dataset.classes.unique())
        model_hparams = hparams["model_hparams"]
        classifier_type = hparams["classifier"]
        model = load_model(
            model_name,
            classifier_type,
            models_path,
            model_hparams,
            d_steps_per_sequence,
            h_steps_per_sequence,
            t_steps_per_sequence,
            num_classes,
        )
        log.info(f"Loaded model: \n {model}")
        sys.stdout.flush()
        test_loader = DataLoader(
            test_dataset, batch_size=model_hparams["batch_size"], shuffle=True
        )
        raw_preds, test_labels = test_model(
            model,
            test_loader
        )
        sys.stdout.flush()
        pred_labels = np.argmax(raw_preds, axis=1)
        contingency_matrix(
            pred_labels, test_labels, plots_path, experiment_name
        )
        sys.stdout.flush()
        np_table, table, table_normalized = build_cross_table(
            pred_labels, test_labels
        )
        acc, prec, rec, f1, _ = compute_classification_metrics(
            np_table, mode="macro"
        )
        roc_auc = compute_roc_auc(raw_preds, test_labels)
        scores = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "execution_time": -1,
            "f1_score": f1,
            "roc_auc_score": roc_auc,
        }
        results_file_path = results_path / "test_results.csv"
        save_experiments_results(
            results_file_path, experiment_name, scores, hparams
        )
