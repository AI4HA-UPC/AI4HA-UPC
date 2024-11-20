import logging
import sys
import time
from pathlib import Path

import torch
import yaml

from datasets.loaders_eeg import get_dataset
from models.classifiers.models_eeg import ResnetSignalClassificationEEG
from utils.dl_utils import EarlyStop, train_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics,
)
from utils.plots import train_loss_plots

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("configurations/eeg_hparams_train.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/eeg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/eeg")
)

EARLY_STOP = EarlyStop(0.001, 3)

if __name__ == "__main__":
    log.info(f"Train of EEG models for AI4HA")
    models_path = ARTIFACTS_PATH / "models"
    # models_path.mkdir(exist_ok=True)
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        log.info(f"Starting experiment {experiment_name}")
        dataset = get_dataset(hparams)
        if len(dataset) == 2:
            train_dataset = dataset[0]
            test_dataset = dataset[1]
            validation_dataset = None
        elif len(dataset) == 3:
            train_dataset = dataset[0]
            validation_dataset = dataset[1]
            test_dataset = dataset[2]
        else:
            train_dataset = dataset
            test_dataset = None
            validation_dataset = None

        num_observations = len(train_dataset)
        num_classes = len(dataset.classes.unique())
        log.info(
            f"Example of the first element of the data: {train_dataset[0]} "
            f"\n Num classes: {num_classes}"
            f"\n Class representation: {train_dataset.get_class_representation()}"
            f"\n Number of observations: {num_observations}"
            f"\n Columns for each observation: {train_dataset.features.shape[-1]}"
        )
        sys.stdout.flush()
        sys.stdout.flush()

        t_steps_per_sequence = train_dataset.get_steps_per_sequence()
        split = 0.9
        hparams["dataset_size"] = len(train_dataset)
        model = ResnetSignalClassificationEEG(
            t_steps_per_sequence,
            hparams["num_conv_blocks"],
            hparams["n_neurons"],
            num_classes=num_classes,
        )
        log.info(f"The model: \n{model}")
        sys.stdout.flush()
        start_time = time.time()
        results = train_model(
            model,
            train_dataset,
            validation_dataset,
            test_dataset,
            hparams,
            experiment_name,
            models_path,
            split=split,
        )
        end_time = time.time()
        plots_path = ARTIFACTS_PATH / "plots"
        train_loss_plots(results, experiment_name, plots_path)
        contingency_matrix(
            results["predictions_and_targets"], plots_path, experiment_name
        )
        execution_time = round(end_time - start_time, 3)
        log.warning(f"Total execution time: {execution_time} seconds")
        sys.stdout.flush()
        np_table, table, table_normalized = build_cross_table(
            results["predictions_and_targets"]
        )
        acc, prec, rec, f1, _ = compute_classification_metrics(np_table, mode="macro")
        scores = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "execution_time": execution_time,
        }
        results_file_path = ARTIFACTS_PATH / "results/results.csv"
        save_experiments_results(results_file_path, experiment_name, scores, hparams)
    exit(0)
