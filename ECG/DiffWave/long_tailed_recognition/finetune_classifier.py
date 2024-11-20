import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.loaders_ecg import get_dataset
from long_tailed_recognition.features_datasets.loaders_ecg_features import FeaturesDataset
from models.classifiers.custom_conv1d import Resnet1DSignalClassification
from models.classifiers.custom_conv1d_classification_head import Resnet1DSignalClassificationHead
from models.classifiers.custom_conv1d_feature_extractor import (
    Resnet1DSignalClassificationFeatureExtractor,
)
from models.classifiers.transformer import TransformerClassifier
from models.classifiers.transformer_feature_extractor import TransformerClassifierFeatureExtractor
from utils.dl_utils import extract_features, train_model, test_model, long_tail_fine_tune_model
from utils.experiments_utils import save_experiments_results
from utils.logging_configuration import log_handler
from utils.metrics import contingency_matrix, build_cross_table, compute_classification_metrics, compute_roc_auc
from utils.plots import train_loss_plots

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("configurations/ecg_finetune_classifier.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc070582/artifacts/ecg")
)


def load_features_data(folder_path: Path, file_name: str, subset_classes: list = None):
    file_path = folder_path / file_name
    data = []
    labels = []
    if file_path.suffix == ".npy":
        with open(file_path, "rb") as fp:
            np_data = np.load(fp)
        lbls = np_data[:, -1]
        data.append(np_data[:, :-1])
        labels.append(lbls)
    elif file_path.suffix == ".npz":
        np_data = np.load(file_path)
        features = np_data.get("samples")
        lbls = np_data.get("classes")
        if subset_classes is not None:
            mask= np.where(np.isin(lbls, subset_classes))
            lbls = lbls[mask]
            features = features[mask]
        data.append(features.squeeze())
        labels.append(lbls)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    idx = [i for i, _ in enumerate(labels)]
    try:
        with open(features_path / f"val_indices_{file_name}.npy", "rb") as fp:
            val_idx = np.load(fp)
    except FileNotFoundError:
        val_idx = np.random.choice(idx, replace=False, size=int(len(idx) * 0.1))
        with open(features_path / f"val_indices_{file_name}.npy", "wb") as fp:
            np.save(fp, val_idx)
    validation_data = data[val_idx]
    validation_labels = labels[val_idx]
    train_idx = np.array(list(set(idx).difference(set(val_idx))))
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    train_ds = FeaturesDataset(train_data, train_labels)
    validation_ds = FeaturesDataset(validation_data, validation_labels)

    return train_ds, validation_ds


if __name__ == "__main__":
    models_path = ARTIFACTS_PATH / "models"
    features_path = ARTIFACTS_PATH / "long_tailed_recognition"
    results_path = ARTIFACTS_PATH / "results"
    plots_path = ARTIFACTS_PATH / "plots"

    features_path.mkdir(exist_ok=True)
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        train_dataset, validation_dataset, test_dataset = get_dataset(hparams, "TrRTeR")
        if hparams["standardize"]:
            validation_dataset.standardize_data(
                train_dataset.means, train_dataset.stds
            )
            test_dataset.standardize_data(
                train_dataset.means, train_dataset.stds
            )

        for synthetic_features_file in hparams["synthetic_features_files"]:
            file_name = synthetic_features_file.split(".")[0]
            experiment_file_name = experiment_name + f"_{file_name}"
            log.info(f"Starting experiment {experiment_file_name}")
            subexperiment_name = experiment_file_name + "_classification_head"
            folder = features_path / hparams["folder"]
            subset_cls = hparams.get("subset_classes")
            train_real_features_dataset, validation_real_features_dataset = load_features_data(folder, hparams["real_features_file"])
            train_synthetic_features_dataset, validation_synthetic_features_dataset = load_features_data(folder, synthetic_features_file, subset_cls)

            t_steps_per_sequence = train_dataset.get_steps_per_sequence()
            h_steps_per_sequence = train_dataset.get_height_per_sequence()
            d_steps_per_sequence = train_dataset.get_depth_per_sequence()
            num_observations = len(train_real_features_dataset.features) + len(train_synthetic_features_dataset.features)
            hparams["dataset_size"] = num_observations
            num_classes = len(train_real_features_dataset.get_class_representation()[0])
            model_hparams = hparams.get("model_hparams")
            model_name = hparams["model"]

            if hparams["classifier"] == "custom_1dcnn_classification_head":
                original_model = Resnet1DSignalClassification(
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
                classifier = Resnet1DSignalClassificationHead(
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
                classifier.classifier_layers.load_state_dict(original_model.classifier_layers.state_dict())

            else:
                log.error("No pretrained classifier specified in the configurations.")
                exit(1)

            log.info(f"Loaded model: \n {classifier}")
            sys.stdout.flush()
            start_time = time.time()
            results = long_tail_fine_tune_model(
                classifier,
                train_real_features_dataset,
                validation_real_features_dataset,
                validation_real_features_dataset,
                train_synthetic_features_dataset,
                validation_synthetic_features_dataset,
                validation_synthetic_features_dataset,
                model_hparams,
                subexperiment_name,
                models_path,
            )
            end_time = time.time()
            train_loss_plots(results, subexperiment_name, plots_path)
            raw_preds = results["predictions_and_targets"][0]
            labels = results["predictions_and_targets"][1]
            pred_labels = np.argmax(raw_preds, axis=1)
            contingency_matrix(
                pred_labels, labels, plots_path, subexperiment_name
            )

            if hparams["classifier"] == "custom_1dcnn_classification_head":
                classifier.load_state_dict(torch.load(models_path / f"{subexperiment_name}.pt"))
                original_model.classifier_layers.load_state_dict(classifier.classifier_layers.state_dict())
            else:
                log.error("No pretrained classifier specified in the configurations.")
                exit(1)
            subexperiment_name = experiment_file_name + "_final_model"
            test_loader = DataLoader(test_dataset, batch_size=model_hparams["batch_size"], shuffle=True)
            raw_preds, test_labels = test_model(
                original_model,
                test_loader
            )
            sys.stdout.flush()
            pred_labels = np.argmax(raw_preds, axis=1)
            contingency_matrix(
                pred_labels, test_labels, plots_path, subexperiment_name
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
            results_file_path = results_path / "test_results_long_tail.csv"
            save_experiments_results(
                results_file_path, subexperiment_name, scores, hparams
            )

            torch.save(original_model.state_dict(), models_path / (subexperiment_name + ".pt"))

    exit(0)
