import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.loaders_ecg import get_dataset
from models.classifiers.custom_conv1d import Resnet1DSignalClassification
from models.classifiers.custom_conv1d_feature_extractor import (
    Resnet1DSignalClassificationFeatureExtractor,
)
from models.classifiers.transformer import TransformerClassifier
from models.classifiers.transformer_feature_extractor import TransformerClassifierFeatureExtractor
from utils.dl_utils import extract_features
from utils.logging_configuration import log_handler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("../configurations/ecg_feature_extractor.yml") as fp:
    EXPERIMENTS = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../../artifacts/ecg")
    if Path("../../artifacts").exists()
    else Path("/home/bsc70/bsc070582/artifacts/ecg")
)

if __name__ == "__main__":
    models_path = ARTIFACTS_PATH / "models"
    features_path = ARTIFACTS_PATH / "features"

    features_path.mkdir(exist_ok=True)
    for experiment_name, hparams in EXPERIMENTS["experiments"].items():
        train_dataset, _, _ = get_dataset(hparams, "TrRTeR")

        t_steps_per_sequence = train_dataset.get_steps_per_sequence()
        h_steps_per_sequence = train_dataset.get_height_per_sequence()
        d_steps_per_sequence = train_dataset.get_depth_per_sequence()
        hparams["dataset_size"] = len(train_dataset)
        num_observations = len(train_dataset)
        num_classes = len(train_dataset.get_class_representation()[0])
        model_hparams = hparams.get("model_hparams")
        model_name = hparams["model"]

        if hparams["classifier"] == "custom_1dcnn_feature_extractor":
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
            model = Resnet1DSignalClassificationFeatureExtractor(
                data_d_steps=d_steps_per_sequence,
                data_h_steps=h_steps_per_sequence,
                data_t_steps=t_steps_per_sequence,
                num_conv_blocks=model_hparams["n_conv_blocks"],
                filter_size=model_hparams["filter_size"],
                n_kernels=model_hparams["n_kernels"],
                num_classes=num_classes,
            )
            model.input.load_state_dict(original_model.input.state_dict())
            model.conv_blocks.load_state_dict(original_model.conv_blocks.state_dict())
        if hparams["classifier"] == "transformer_feature_extractor":
            original_model = TransformerClassifier(
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
            model = TransformerClassifierFeatureExtractor(
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
            model.patchify.load_state_dict(original_model.patchify.state_dict())
            model.pos_embedding.load_state_dict(original_model.pos_embedding.state_dict())
            model.blocks.load_state_dict(original_model.blocks.state_dict())

        log.info(f"Loaded model: \n {model}")
        model.freeze_all()
        sys.stdout.flush()
        start_time = time.time()
        features = extract_features(model, train_dataset, model_hparams)
        with open(features_path / f"{model_name}_features.npy", mode="wb") as fp:
            np.save(fp, features)

    exit(0)
