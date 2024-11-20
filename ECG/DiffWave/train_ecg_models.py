import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.loaders_ecg import get_dataset, DATASET_SIZES
from models.classifiers.custom_conv1d import Resnet1DSignalClassification
from models.classifiers.transformer import TransformerClassifier
from utils.dl_utils import EarlyStop, train_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics,
    compute_roc_auc,
)
from utils.plots import train_loss_plots
import torch
seed = 1704
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("configurations/ecg_hparams_train.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)

EARLY_STOP = EarlyStop(0.001, 3)
NUM_REPETITIONS = 20 #
"""
TODO Experiments with synthetic data:
    1. Classification of the synthetic data as test set.
    2. Data augmentation for less representative classes.
    3. Transfer learning: train on synthetic and then use real.
"""

if __name__ == "__main__":
    log.info(f"Train of ECG models for AI4HA")
    models_path = ARTIFACTS_PATH / "models"
    results_file_path = ARTIFACTS_PATH / "results/results.csv"
    # models_path.mkdir(exist_ok=True)
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        for experiment_setting in hparams["experiment_settings"]:
            experiment_name_setting = f"{experiment_name}_{experiment_setting}"
            log.info(f"Starting experiment {experiment_name_setting}")
            standardize = hparams.get("standardize", False)
            hparams["standardize"] = False
            train_dataset, validation_dataset, test_dataset = get_dataset(
                hparams, experiment_setting
            )
            hparams["standardize"] = standardize
            for size_name, size_num in DATASET_SIZES._asdict().items():
                if size_num != 1:
                    reduced_train_dataset = train_dataset.copy_dataset(
                        dataset_size=size_num
                    )
                    reduced_validation_dataset = validation_dataset.copy_dataset()
                    reduced_test_dataset = test_dataset.copy_dataset()
                else:
                    reduced_train_dataset = train_dataset
                    reduced_validation_dataset = validation_dataset
                    reduced_test_dataset = test_dataset

                if hparams["standardize"]:
                    reduced_train_dataset.compute_standardize_parameters()
                    reduced_train_dataset.standardize_data(
                        reduced_train_dataset.means, reduced_train_dataset.stds
                    )
                    reduced_validation_dataset.standardize_data(
                        reduced_train_dataset.means, reduced_train_dataset.stds
                    )
                    reduced_test_dataset.standardize_data(
                        reduced_train_dataset.means, reduced_train_dataset.stds
                    )
                experiment_name_full = f"{experiment_name_setting}_{size_name}_{reduced_train_dataset.dataset_name()}"

                num_observations = len(reduced_train_dataset)
                num_classes = len(reduced_train_dataset.get_class_representation()[0])

                classes_to_remove = hparams.get("remove_classes")
                if classes_to_remove:
                    reduced_train_dataset.remove_classes(classes_to_remove)
                    test_dataset.remove_classes(classes_to_remove)

                log.info(
                    f"Example of the first element of the data: {reduced_train_dataset[0]} "
                    f"\n Num classes: {num_classes}"
                    f"\n Class representation: {reduced_train_dataset.get_class_representation()}"
                    f"\n Number of observations: {num_observations}"
                    f"\n Columns for each observation: {reduced_train_dataset.features.shape[-1]}"
                )
                sys.stdout.flush()
                sys.stdout.flush()

                t_steps_per_sequence = reduced_train_dataset.get_steps_per_sequence()
                h_steps_per_sequence = reduced_train_dataset.get_height_per_sequence()
                d_steps_per_sequence = reduced_train_dataset.get_depth_per_sequence()
                hparams["dataset_size"] = len(reduced_train_dataset)
                generative_model = hparams.get("generative_model")
                model_hparams = hparams.get("model_hparams")

                scores = {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "execution_time": [],
                    "f1_score": [],
                    "roc_auc_score": [],
                }
                experiment_name_full_run = experiment_name_full
                for run in range(NUM_REPETITIONS):
                    if hparams.get("classifier") == "transformer":
                        model = TransformerClassifier(
                            seq_length=t_steps_per_sequence,
                            seq_channels=d_steps_per_sequence,
                            n_classes=num_classes,
                            patch_size=model_hparams["patch_size"],
                            data_embed_dim=model_hparams["data_embed_dim"],
                            n_layers=model_hparams["n_layers"],
                            n_heads=model_hparams["n_heads"],
                            dropout_rate=model_hparams["dropout_rate"],
                            class_logits=model_hparams["class_logits"],
                        )
                        log.warning(
                            f"Total num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
                        )
                    else:
                        model = Resnet1DSignalClassification(
                            data_d_steps=d_steps_per_sequence,
                            data_h_steps=h_steps_per_sequence,
                            data_t_steps=t_steps_per_sequence,
                            num_conv_blocks=model_hparams["n_conv_blocks"],
                            num_layers_classifier=model_hparams["n_layers_classifier"],
                            n_neurons_classifier=model_hparams["n_neurons_classifier"],
                            filter_size=model_hparams["filter_size"],
                            n_kernels=model_hparams["n_kernels"],
                            num_classes=num_classes,
                        )
                    log.info(f"The model: \n{model}")
                    sys.stdout.flush()

                    log.info(
                        f"Experiment {experiment_name_full} - Iteration size - {size_name} - Run: {run}"
                    )
                    experiment_name_full_run = f"{experiment_name_full}_{run}"
                    sys.stdout.flush()
                    start_time = time.time()
                    results = train_model(
                        model,
                        reduced_train_dataset,
                        reduced_validation_dataset,
                        reduced_test_dataset,
                        model_hparams,
                        experiment_name_full_run,
                        models_path,
                    )
                    end_time = time.time()
                    plots_path = ARTIFACTS_PATH / "plots"
                    train_loss_plots(results, experiment_name_full_run, plots_path)
                    raw_preds = results["predictions_and_targets"][0]
                    labels = results["predictions_and_targets"][1]
                    pred_labels = np.argmax(raw_preds, axis=1)
                    contingency_matrix(
                        pred_labels, labels, plots_path, experiment_name_full_run
                    )
                    execution_time = round(end_time - start_time, 3)
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
                    log.warning(f"Argument with best f1 score: {np.argmax(scores['f1_score'])}")
                scores = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
                save_experiments_results(
                    results_file_path, experiment_name_full_run, scores, hparams
                )
    exit(0)
