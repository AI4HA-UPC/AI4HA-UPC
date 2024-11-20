import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from einops import rearrange, repeat, reduce
from optuna.trial import TrialState
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from datasets.loaders_ecg import get_dataset
from models.classifiers.layers.Layers import PositionalEncoding, Patchify
from utils.dl_utils import train_model
from utils.logging_configuration import log_handler
from utils.metrics import (
    build_cross_table,
    compute_classification_metrics,
    compute_roc_auc,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

with open("configurations/ecg_hparams_random_search.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
CLASSES = 7
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)


class TransformerClassifier(nn.Module):
    """_TransformerClassifier model_

    Mostly copied from TTSGANDiscriminator

    class_logits = classtoken: only the class token is used for
                   classification
                 = avgpool: average pooling
    """

    def __init__(
        self,
        trial: optuna.Trial,
        seq_length=512,
        seq_channels=3,
        n_classes=9,
        patch_size=100,
        data_embed_dim=128,
        n_layers=8,
        num_heads=16,
        dropout_rate=0.25,
        pos_encodings=True,
        class_logits="avgpool",
    ):
        super(TransformerClassifier, self).__init__()
        self.seq_len = seq_length
        self.seq_channels = seq_channels
        self.n_classes = n_classes
        self.patch_size = trial.suggest_int("patch_size", 5, 500)
        data_embed_dim_power = trial.suggest_int("data_embed_power", 5, 9)
        self.data_embed_dim = 2**data_embed_dim_power
        self.n_layers = trial.suggest_int("n_layers", 2, 16)
        n_heads_power = trial.suggest_int("n_heads_power", 4, data_embed_dim_power)
        self.n_heads = 2**n_heads_power
        self.dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)
        self.pos_encodings = pos_encodings
        self.class_logits = trial.suggest_categorical(
            "class_logits", ["classtoken", "avgpool"]
        )

        # Reduces the input time series into a set of tokens
        self.patchify = Patchify(
            self.seq_channels, self.data_embed_dim, self.patch_size
        )
        # Adds positional encoding to the tokens
        self.pos_embedding = PositionalEncoding(self.data_embed_dim, self.dropout_rate)
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.data_embed_dim))

        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.data_embed_dim,
            nhead=self.n_heads,
            activation=nn.GELU(),
            norm_first=True,
            dropout=self.dropout_rate,
        )

        # Transformer encoder
        self.blocks = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.n_layers
        )

        # Classification head (logits)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.data_embed_dim),
            nn.Linear(self.data_embed_dim, n_classes),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        batch = x.shape[0]
        # convert to tokens
        x = self.patchify(x)
        x = rearrange(x, "b e t -> b t e")

        # add class token
        cls_tokens = repeat(self.cls_token, "() t e -> b t e", b=batch)

        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding
        x = rearrange(x, "b t e -> t b e")
        if self.pos_encodings:
            x = self.pos_embedding(x)
        # transformer
        x = self.blocks(x)

        # Only the class token is used for classification
        if self.class_logits == "classtoken":
            x = x[0]
        # Average pooling
        elif self.class_logits == "avgpool":
            x = reduce(x, "t b e -> b e", reduction="mean")

        logit = self.class_head(x)
        return logit


class RSResidual1DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: int):
        super(RSResidual1DBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv1d(
            self.num_neurons, self.num_neurons, filter_size, padding="same", dilation=5
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class RSConvolutional1DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: int):
        super(RSConvolutional1DBlock, self).__init__()
        self.residual_block = RSResidual1DBlock(num_neurons, filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(5, stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class RSResnet1DSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        trial: optuna.Trial,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        num_kernels: int = 8,
        num_neurons: int = 32,
        num_classes: int = 5,
        filter_size: int = 5,
        dropout_rate: float = 0.25,
    ):
        super(RSResnet1DSignalClassification, self).__init__()

        self.num_conv_blocks = trial.suggest_int("n_conv_blocks", 1, 7)
        self.num_kernels = trial.suggest_categorical(
            "n_kernels", [2**pow for pow in range(2, 6)]
        )
        self.filter_size = trial.suggest_categorical("filter_size", [3, 5, 7, 9])
        self.input = nn.Conv1d(
            data_d_steps, self.num_kernels, self.filter_size, padding="same"
        )
        for _ in range(self.num_conv_blocks):
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_t_steps
        self.conv_blocks = nn.ModuleList(
            [
                RSConvolutional1DBlock(self.num_kernels, self.filter_size)
                for _ in range(self.num_conv_blocks)
            ]
        )
        n_layers = trial.suggest_int("n_layers", 1, 3)
        in_features = self.num_kernels * data_total_steps
        layers = []
        for num_layers in range(n_layers):
            out_features = trial.suggest_categorical(
                f"n_units_fc_l_{num_layers}", [2**p for p in range(2, 10)]
            )
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float(f"drouput_fc_l_{num_layers}", 0, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))
        self.classifier_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, time = x.shape
        x = x.view(bsz, -1)
        y = self.classifier_layers(x)

        return y


class RSResidual2DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: Tuple[int, int]):
        super(RSResidual2DBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv2d(
            self.num_neurons, self.num_neurons, filter_size, padding="same", dilation=5
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class RSConvolutional2DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: Tuple[int, int]):
        super(RSConvolutional2DBlock, self).__init__()
        self.residual_block = RSResidual2DBlock(num_neurons, filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((2, 5), stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class RSResnet2DSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        trial: optuna.Trial,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        num_neurons: int = 32,
        num_classes: int = 5,
        filter_size: Tuple[int, int] = (5, 5),
        dropout_rate: float = 0.25,
    ):
        super(RSResnet2DSignalClassification, self).__init__()

        self.num_conv_blocks = trial.suggest_int("n_conv_blocks", 1, 3)
        self.num_neurons = trial.suggest_categorical(
            "n_kernels", [2**i for i in range(2, 6)]
        )
        for i in range(num_conv_blocks):
            data_d_steps = int((data_d_steps - 2) / 2) + 1
            data_h_steps = int((data_h_steps - 2) / 2) + 1
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_d_steps * data_h_steps * data_t_steps

        self.input = nn.Conv2d(1, self.num_neurons, filter_size, padding="same")
        self.conv_blocks = nn.ModuleList(
            [
                RSConvolutional2DBlock(self.num_neurons, filter_size)
                for _ in range(num_conv_blocks)
            ]
        )
        n_layers = trial.suggest_int("n_layers", 1, 3)
        in_features = self.num_neurons * data_total_steps
        layers = []
        for i in range(n_layers):
            out_features = trial.suggest_categorical(
                f"n_units_fc_l_{i}", [2**p for p in range(2, 10)]
            )
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float(f"drouput_fc_l_{i}", 0, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))
        layers.append(nn.LogSoftmax(dim=-1))
        self.classifier_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, h, time = x.shape
        x = x.view(bsz, -1)
        y = self.classifier_layers(x)
        return y


def objective(trial, dataset, hparams):
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
    w_steps_per_sequence = train_dataset.get_steps_per_sequence()
    h_steps_per_sequence = train_dataset.get_height_per_sequence()
    d_steps_per_sequence = train_dataset.get_depth_per_sequence()
    if hparams["classifier"] == "transformer":
        model = TransformerClassifier(
            trial,
            seq_length=w_steps_per_sequence,
            seq_channels=d_steps_per_sequence,
            n_classes=7,
        )
    else:
        model = RSResnet1DSignalClassification(
            trial,
            d_steps_per_sequence,
            h_steps_per_sequence,
            w_steps_per_sequence,
            num_classes=7,
        )
    log.info(f"The model: \n{model}")

    hparams["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    results = train_model(
        model,
        train_dataset,
        validation_dataset,
        test_dataset,
        hparams,
        experiment_name,
        save_model=False,
        save_results=False,
    )
    raw_preds = results["predictions_and_targets"][0]
    labels = results["predictions_and_targets"][1]
    pred_labels = np.argmax(raw_preds, axis=1)
    np_table, table, table_normalized = build_cross_table(pred_labels, labels)
    acc, prec, rec, f1, _ = compute_classification_metrics(np_table, mode="macro")
    try:
        macro_auc = compute_roc_auc(raw_preds, labels)
    except ValueError as e:
        log.error(f"0 roc score because {e}")
        macro_auc = 0
    return f1, macro_auc, acc


if __name__ == "__main__":
    plots_path = ARTIFACTS_PATH / "plots"

    objectives = ["f1", "macro_auc", "acc"]
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        dataset = get_dataset(hparams, hparams.get("experiment_settings"))
        log.info(f"Starting experiment {experiment_name}")
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            storage="sqlite:///db.sqlite3",
            study_name=experiment_name,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(
                trial,
                dataset,
                hparams,
            ),
            n_trials=hparams["trials"],
        )

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        log.info("Study statistics: ")
        log.info(f"  Number of finished trials: {len(study.trials)}")
        log.info(
            f"  Number of pruned trials: {len(pruned_trials)}",
        )
        log.info(
            f"  Number of complete trials: {len(complete_trials)}",
        )

        log.info("Best trial:")
        trials = study.best_trials

        for trial in trials:
            log.info(f"  Value: { trial.values}")

            log.info("  Params: ")
            for key, value in trial.params.items():
                log.info(f"    {key}: {value}")

        for i, metric in enumerate(objectives[:-1]):
            best = max(study.best_trials, key=lambda t: t.values[i])
            log.info(f"Trial with max {metric}")
            log.info("  Params: ")
            for key, value in trial.params.items():
                log.info(f"    {key}: {value}")

        # TODO: For the future

        #     vis.plot_contour(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_contour_{metric}_{experiment_name}.png")
        #
        #     vis.plot_parallel_coordinate(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_parallel_coordinate_{metric}_{experiment_name}.png")
        #     vis.plot_optimization_history(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_history_{metric}_{experiment_name}.png")
        #     vis.plot_edf(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_param_distribution_{metric}_{experiment_name}.png")
        #     vis.plot_slice(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_slice_{metric}_{experiment_name}.png")
        #     vis.plot_rank(study, target=lambda t: t.values[i])
        #     plt.savefig(plots_path / f"optuna_rank_{metric}_{experiment_name}.png")
        #     # vis.plot_contour(study, params=["n_conv_blocks", "n_kernels"], target=lambda t: t.values[i])
        #     # plt.savefig(plots_path / f"optuna_contour_{metric}_n_layers_vs_n_conv_blocks_{experiment_name}.png")
        #     # vis.plot_contour(study, params=["n_layers", "n_conv_blocks"], target=lambda t: t.values[i])
        #     # plt.savefig(plots_path / f"optuna_contour_{metric}_classifier_neurons_vs_kernels_{experiment_name}.png")
        # vis.plot_intermediate_values(study)
        # plt.savefig(plots_path / f"optuna_intermediate_values_{experiment_name}.png")
        # vis.plot_timeline(study)
        # plt.savefig(plots_path / f"optuna_timeline_{experiment_name}.png")
        # vis.plot_param_importances(study)
        # plt.savefig(plots_path / f"optuna_param_importances_{experiment_name}.png")
        # vis.plot_param_importances(
        #     study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        # )
        # plt.savefig(plots_path / f"optuna_param_importances_duration_{experiment_name}.png")
