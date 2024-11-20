import argparse
from pathlib import Path
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import pandas as pd
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
from geomloss import SamplesLoss
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("configurations/ecg_two_sample_test.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)

EARLY_STOP = EarlyStop(0.001, 3)
NUM_REPETITIONS = 1


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


def two_sample_test_score(acc: float):
    n_acc = acc / 100
    return 1 - np.abs(n_acc - 0.5) / 0.5


if __name__ == "__main__":
    log.info(f"Two sample classifier test")
    results_file_path = ARTIFACTS_PATH / "results/two_sample_test_results.csv"
    plots_path = ARTIFACTS_PATH / "plots"

    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        for experiment_setting in hparams["experiment_settings"]:
            experiment_name_setting = f"{experiment_name}_{experiment_setting}"
            standardize = hparams.get("standardize", False)
            hparams["standardize"] = False
            train_dataset, validation_dataset, test_dataset = get_dataset(
                hparams, experiment_setting
            )
            hparams["standardize"] = standardize

            t_steps_per_sequence = train_dataset.get_steps_per_sequence()
            h_steps_per_sequence = train_dataset.get_height_per_sequence()
            d_steps_per_sequence = train_dataset.get_depth_per_sequence()
            generative_model = hparams.get("generative_model")
            model_hparams = hparams.get("model_hparams")

            num_classes = len(train_dataset.get_class_representation()[0])
            array_num_classes = [
                i[0] for i in enumerate(train_dataset.original_classes.unique())
            ]
            array_num_classes = array_num_classes + ["all"]
            for cl in array_num_classes:
                cl_train_dataset = train_dataset.copy_dataset()
                cl_val_dataset = validation_dataset.copy_dataset()
                cl_test_dataset = test_dataset.copy_dataset()
                if cl != "all":
                    cl_train_dataset.specific_class(cl)
                    cl_val_dataset.specific_class(cl)
                    cl_test_dataset.specific_class(cl)

                if hparams["standardize"]:
                    cl_train_dataset.compute_standardize_parameters()
                    cl_train_dataset.standardize_data(
                        cl_train_dataset.means, cl_train_dataset.stds
                    )
                    cl_val_dataset.standardize_data(
                        cl_train_dataset.means, cl_train_dataset.stds
                    )
                    cl_test_dataset.standardize_data(
                        cl_train_dataset.means, cl_train_dataset.stds
                    )

                hparams["dataset_size"] = len(train_dataset)
                num_observations = len(train_dataset)
                experiment_name_full = f"{experiment_name_setting}_{train_dataset.dataset_name()}_class_{cl}"

                # TODO: Implement in another script
                # for ch in range(12):
                #     l = SamplesLoss("gaussian", blur=0.1)
                #     min_len = torch.min([len(train_dataset.features), len(validation_dataset.features, len(test_dataset.features))])
                #     mmd_train_validation = l(train_dataset.features[:min_len,ch].T, validation_dataset.features[:min_len,ch].T)
                #     mmd_train_test = l(train_dataset.features[:min_len,ch].T, test_dataset.features[:min_len,ch].T)

                scores = {"accuracy": [], "precision": []}
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

                    results = train_model(
                        model,
                        train_dataset,
                        validation_dataset,
                        test_dataset,
                        model_hparams,
                        save_model=False,
                    )
                    raw_preds = results["predictions_and_targets"][0]
                    labels = results["predictions_and_targets"][1]
                    pred_labels = np.argmax(raw_preds, axis=1)
                    np_table, table, table_normalized = build_cross_table(
                        pred_labels, labels
                    )
                    acc, prec, rec, f1, _ = compute_classification_metrics(
                        np_table, mode="macro"
                    )
                    tsts = two_sample_test_score(acc)
                    scores["accuracy"].append(acc)
                    scores["precision"].append(tsts)
                    contingency_matrix(
                        pred_labels, labels, plots_path, experiment_name_full
                    )
                scores = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
                save_experiments_results(
                    results_file_path, experiment_name_full, scores, hparams
                )
