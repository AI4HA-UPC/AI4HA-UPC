import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from datasets.loaders_ecg import get_dataset, DATASET_SIZES
from models.classifiers.models_ecg import ResnetSignalClassification
from utils.dl_utils import load_model, train_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import (
    contingency_matrix,
    build_cross_table,
    compute_classification_metrics,
)
from utils.plots import train_loss_plots
from models.classifiers.transformer import TransformerClassifier
from models.classifiers.custom_conv1d import Resnet1DSignalClassification

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
NUM_REPETITIONS = 10

if __name__ == "__main__":
    log.info("Start transfer learning experiments")
    sys.stdout.flush()
    models_path = ARTIFACTS_PATH / "models"
    plots_path = ARTIFACTS_PATH / "plots"
    results_file_path = ARTIFACTS_PATH / "results/tf_results.csv"
    for experiment, hparams in EXPERIMENTS_TF_FT["experiments"].items():
        model_name = hparams["model"]
        train_dataset, validation_dataset, test_dataset = get_dataset(hparams, "TrSTeR")
        for size_name, size_num in DATASET_SIZES._asdict().items():
            hparams["dataset_size"] = size_num

            reduced_test_dataset = test_dataset.copy_dataset(dataset_size=size_num)

            train_dataset.compute_standardize_parameters()

            if hparams["standardize"]:
                validation_dataset.standardize_data(
                    train_dataset.means, train_dataset.stds
                )
                reduced_test_dataset.standardize_data(
                    train_dataset.means, train_dataset.stds
                )

            t_steps_per_sequence = train_dataset.get_steps_per_sequence()
            h_steps_per_sequence = train_dataset.get_height_per_sequence()
            d_steps_per_sequence = train_dataset.get_depth_per_sequence()
            hparams["dataset_size"] = len(train_dataset)
            generative_model = hparams.get("generative_model")
            model_hparams = hparams.get("model_hparams")

            hparams["dataset_size"] = len(train_dataset)
            num_observations = len(train_dataset)
            num_classes = len(train_dataset.get_class_representation()[0])
            experiment_name = (
                f"{experiment}_size_{size_name}_{train_dataset.dataset_name()}"
            )
            scores = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "execution_time": [],
                "f1_score": [],
            }
            for run in range(NUM_REPETITIONS):
                log.info(
                    f"Experiment {experiment_name} - Iteration size - {size_name} - Run: {run}"
                )
                sys.stdout.flush()

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
                elif hparams.get("classifier") == "custom_1dcnn":
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

                if model_name == "new":
                    model = ResnetSignalClassification(
                        t_steps_per_sequence,
                        hparams["num_conv_blocks"],
                        hparams["n_neurons"],
                        num_classes=num_classes,
                    )
                else:
                    model = load_model(
                        model_name,
                        "ResnetSignalClassificationEEG",
                        models_path,
                        hparams,
                        1,
                        1,
                        t_steps_per_sequence,
                        num_classes,
                    )
                if hparams["dataset"] == "ptb-diagnostic":
                    num_features = model.linear3.in_features
                    model.linear3 = nn.Linear(num_features, 2)
                log.info(f"Loaded model: \n {model}")
                if hparams["freeze_layers"]:
                    model.freeze_all_but_last()
                sys.stdout.flush()
                start_time = time.time()
                results = train_model(
                    model,
                    train_dataset,
                    validation_dataset,
                    reduced_test_dataset,
                    hparams,
                    experiment_name,
                    models_path,
                    results_path=ARTIFACTS_PATH,
                    save_model=True,
                    save_results=True,
                )
                end_time = time.time()
                execution_time = end_time - start_time

                train_loss_plots(results, experiment_name, plots_path)
                contingency_matrix(
                    results["predictions_and_targets"], plots_path, experiment_name
                )
                log.info(f"Total execution time: {execution_time} seconds")
                np_table, table, table_normalized = build_cross_table(
                    results["predictions_and_targets"]
                )
                acc, prec, rec, f1, _ = compute_classification_metrics(
                    np_table, mode="macro"
                )
                scores["accuracy"].append(acc)
                scores["precision"].append(prec)
                scores["recall"].append(rec)
                scores["f1_score"].append(f1)
                scores["execution_time"].append(execution_time)
                log.warning(scores)

            scores = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
            save_experiments_results(
                results_file_path, experiment_name, scores, hparams
            )
    exit(0)
