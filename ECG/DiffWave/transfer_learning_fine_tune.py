import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.loaders_ecg import get_dataset, DATASET_SIZES
from utils.dl_utils import load_model, train_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics,
    compute_roc_auc,
)
from utils.plots import train_loss_plots

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("configurations/ecg_hparams_tf_ft.yml") as fp:
    EXPERIMENTS_TF_FT = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc070582/artifacts/ecg")
)
NUM_REPETITIONS = 25

if __name__ == "__main__":
    log.info("Start transfer learning experiments")
    sys.stdout.flush()
    models_path = ARTIFACTS_PATH / "models"
    plots_path = ARTIFACTS_PATH / "plots"
    results_file_path = ARTIFACTS_PATH / "results/tf_results.csv"
    val_test_standardized = False
    for experiment, hparams in EXPERIMENTS_TF_FT["experiments"].items():
        model_name = hparams["model"]
        train_dataset, validation_dataset, test_dataset = get_dataset(hparams, "TrRTeR")
        for size_name, size_num in DATASET_SIZES._asdict().items():

            reduced_train_dataset = train_dataset.copy_dataset(dataset_size=size_num)

            reduced_train_dataset.compute_standardize_parameters()

            if hparams["standardize"] and not val_test_standardized:
                validation_dataset.standardize_data(
                    reduced_train_dataset.means, reduced_train_dataset.stds
                )
                test_dataset.standardize_data(
                    reduced_train_dataset.means, reduced_train_dataset.stds
                )
                val_test_standardized = True

            t_steps_per_sequence = reduced_train_dataset.get_steps_per_sequence()
            h_steps_per_sequence = reduced_train_dataset.get_height_per_sequence()
            d_steps_per_sequence = reduced_train_dataset.get_depth_per_sequence()
            hparams["dataset_size"] = len(reduced_train_dataset)
            num_observations = len(reduced_train_dataset)
            num_classes = len(reduced_train_dataset.get_class_representation()[0])
            experiment_name = (
                f"{experiment}_size_{size_name}_{reduced_train_dataset.dataset_name()}"
            )
            scores = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "execution_time": [],
                "f1_score": [],
                "roc_auc_score": [],
            }
            model_hparams = hparams.get("model_hparams")

            for run in range(NUM_REPETITIONS):
                log.info(
                    f"Experiment {experiment_name} - Iteration size - {size_name} - Run: {run}"
                )
                sys.stdout.flush()

                model = load_model(
                    model_name,
                    hparams["classifier"],
                    models_path,
                    model_hparams,
                    d_steps_per_sequence,
                    h_steps_per_sequence,
                    t_steps_per_sequence,
                    num_classes,
                )

                log.info(f"Loaded model: \n {model}")
                if hparams["freeze_layers"]:
                    model.freeze_all_but_last()
                sys.stdout.flush()
                start_time = time.time()
                results = train_model(
                    model,
                    reduced_train_dataset,
                    validation_dataset,
                    test_dataset,
                    model_hparams,
                    experiment_name,
                    models_path,
                    results_path=ARTIFACTS_PATH,
                    save_model=True,
                    save_results=True,
                )
                end_time = time.time()
                execution_time = round(end_time - start_time, 3)

                train_loss_plots(results, experiment_name, plots_path)
                raw_preds = results["predictions_and_targets"][0]
                labels = results["predictions_and_targets"][1]
                pred_labels = np.argmax(raw_preds, axis=1)
                contingency_matrix(pred_labels, labels, plots_path, experiment_name)

                log.warning(f"Total execution time: {execution_time} seconds")
                sys.stdout.flush()
                np_table, table, table_normalized = build_cross_table(
                    pred_labels, labels
                )
                acc, prec, rec, f1, _ = compute_classification_metrics(
                    np_table, mode="macro"
                )
                roc_auc = compute_roc_auc(raw_preds, labels)
                scores["accuracy"].append(acc)
                scores["precision"].append(prec)
                scores["recall"].append(rec)
                scores["f1_score"].append(f1)
                scores["roc_auc_score"].append(roc_auc)
                scores["execution_time"].append(execution_time)
                log.warning(scores)

            scores = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
            save_experiments_results(
                results_file_path, experiment_name, scores, hparams
            )
    exit(0)
