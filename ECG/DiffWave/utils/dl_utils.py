import logging
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, random_split

from models.classifiers.custom_conv1d import Resnet1DSignalClassification
from models.classifiers.custom_conv1d_classification_head import Resnet1DSignalClassificationHead
from models.classifiers.models_eeg import ResnetSignalClassificationEEG
from models.classifiers.transformer import TransformerClassifier
from src.utils.logging_configuration import log_handler


class EarlyStop(NamedTuple):
    min_delta: float
    patience: int


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(log_handler)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def correct_predictions(predicted_batch: torch.Tensor, label_batch: torch.Tensor):
    pred = predicted_batch.argmax(dim=1, keepdim=True)
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def train_epoch(
        train_loader: DataLoader,
        network: ResnetSignalClassificationEEG,
        optimizer: Optimizer,
        criterion: CrossEntropyLoss,
        epoch: int,
        class_w: torch.Tensor,
):
    network.to(DEVICE)
    network.train()
    losses = []
    accs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target.squeeze())
        loss.backward()  # hay un problema de indices, que no estan en secuencia luego torch se queja porque hay indice 6 y no hay 6 items en el target.
        acc = 100 * (correct_predictions(output, target) / data.shape[0])
        losses.append(loss.item())
        accs.append(acc)
        optimizer.step()
        if batch_idx % 5 == 0 or batch_idx >= len(train_loader):
            log.info(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAcc: {acc:.1f}"
            )
    mean_losses = np.mean(losses)
    mean_acc = np.mean(accs)
    return mean_losses, mean_acc


def val_epoch(
        val_loader: DataLoader,
        network: ResnetSignalClassificationEEG,
        criterion: CrossEntropyLoss,
):
    network.to(DEVICE)
    network.eval()
    val_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = network(data)
            val_loss = val_loss + criterion(output, target.long()).item()
            acc = acc + correct_predictions(output, target)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * acc / len(val_loader.dataset)
    log.info(
        f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {acc}/{len(val_loader.dataset)} ({val_acc:.0f}%)"
    )
    return val_loss, val_acc


def train_model(
        network: Union[ResnetSignalClassificationEEG, Resnet1DSignalClassification, TransformerClassifier],
        train_ds,
        validation_ds,
        test_ds,
        hparams: dict,
        exp_name: str = "poc_experiment",
        model_path: Path = "",
        split: float = 0.8,
        save_model=True,
        save_results=False,
        results_path=None,
):
    if hparams["weight_classes"]:
        class_weights = torch.Tensor(
            1 - np.unique(train_ds.classes, return_counts=True)[1] / len(train_ds)
        )
        class_weights = class_weights.to(DEVICE)
    else:
        class_weights = None

    early_stop = hparams.get("earlystop")
    if early_stop:
        early_stop = EarlyStop(early_stop["delta"], early_stop["patience"])

    train_loader, validation_loader, test_loader = create_loaders(hparams, train_ds, validation_ds, test_ds)

    network.to(DEVICE)
    num_epochs = hparams["epochs"]
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(network.parameters(), hparams["learning_rate"])

    best_model_state = {}
    best_loss = torch.inf
    not_improved_epochs = 0
    best_loss_epoch = 0
    tr_losses = []
    tr_accuracies = []
    tv_losses = []
    tv_accuracies = []
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            train_loader, network, optimizer, criterion, epoch, class_weights
        )
        tv_loss, tv_acc = val_epoch(validation_loader, network, criterion)
        tv_losses.append(tv_loss)
        tv_accuracies.append(tv_acc)
        tr_losses.append(tr_loss)
        tr_accuracies.append(tr_acc)

        if tv_loss <= (best_loss - early_stop.min_delta):
            best_loss = tv_loss
            best_loss_epoch = epoch
            best_model_state = network.state_dict()
            not_improved_epochs = 0
            log.warning(
                f"Best loss {best_loss} in epoch {best_loss_epoch} from total of {num_epochs + 1} epochs"
            )
        else:
            not_improved_epochs = not_improved_epochs + 1 if hparams["earlystop"] else 0
            if not_improved_epochs == early_stop.patience:
                log.warning(
                    f"{early_stop.patience} without improvements, STOP training \n"
                    f"Best loss {best_loss} in epoch {best_loss_epoch} from total of {num_epochs + 1} epochs"
                )
                break

    predictions_targets = test_model(network, test_loader)
    metrics = {
        "tr_losses": tr_losses,
        "tv_losses": tv_losses,
        "tr_accuracies": tr_accuracies,
        "tv_accuracies": tv_accuracies,
        "predictions_and_targets": predictions_targets,
    }
    if save_results:
        save_model_results(metrics, results_path / (exp_name + ".csv"))
    if save_model:
        torch.save(best_model_state, model_path / (exp_name + ".pt"))

    return metrics


def test_model(network: ResnetSignalClassificationEEG, test_set: DataLoader):
    network.to(DEVICE)
    network.eval()
    raw_predictions = []
    targets = []
    for sample, target in test_set:
        sample, target = sample.to(DEVICE), target.to(DEVICE)
        y_hat = network(sample)
        raw_predictions = raw_predictions + list(y_hat.cpu().detach().numpy())
        targets = targets + list(target.cpu().detach().numpy())
    return raw_predictions, targets


def extract_features(network: ResnetSignalClassificationEEG, dataset, hparams: dict):
    network.to(DEVICE)
    network.eval()
    features = []
    targets = []
    data_loader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=True)

    for sample, target in data_loader:
        sample, target = sample.to(DEVICE), target.to(DEVICE)
        y_hat = network(sample)
        features_with_class = torch.cat([y_hat, target.unsqueeze(1)], dim=1)
        features = features + list(features_with_class.cpu().detach().numpy())
    return np.array(features)


def load_model(
        model_name: str,
        model_class: str,
        models_path: Path,
        model_hparams: dict,
        d_steps_per_sequence: int,
        h_steps_per_sequence: int,
        t_steps_per_sequence: int,
        num_classes: int,
):
    if model_class == "custom_1dcnn":
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
    else:
        model = ResnetSignalClassificationEEG(
            t_steps_per_sequence,
            model_hparams["num_conv_blocks"],
            model_hparams["n_neurons"],
            num_classes=num_classes,
        )
    model.load_state_dict(torch.load(models_path / f"{model_name}.pt"))
    return model


def save_model_results(res: dict, model_results_path: Path):
    pd.DataFrame(
        {
            k: res[k]
            for k in ["tr_losses", "tv_losses", "tr_accuracies", "tv_accuracies"]
        }
    ).to_csv(model_results_path)


def long_tail_fine_tune_loss(y_real_pred, y_real_true, y_synthetic_pred, y_synthetic_true, weight_factor=0.05):
    real_criterion = CrossEntropyLoss()
    synthetic_criterion = CrossEntropyLoss()
    real_loss = real_criterion(y_real_pred, y_real_true)
    synthetic_loss = synthetic_criterion(y_synthetic_pred, y_synthetic_true)
    return weight_factor * synthetic_loss + real_loss

def long_tail_fine_tune_model(
        network: Resnet1DSignalClassificationHead,
        train_real_ds,
        validation_real_ds,
        test_real_ds,
        train_synthetic_ds,
        validation_synthetic_ds,
        test_synthetic_ds,
        hparams: dict,
        exp_name: str = "poc_experiment",
        model_path: Path = "",
        split: float = 0.8,
        save_model=True,
        save_results=False,
        results_path=None,
):
    early_stop = hparams.get("earlystop")
    if early_stop:
        early_stop = EarlyStop(early_stop["delta"], early_stop["patience"])

    train_real_loader, validation_real_loader, test_real_loader = create_loaders(hparams, train_real_ds, validation_real_ds, test_real_ds)
    train_synthetic_loader, validation_synthetic_loader, test_synthetic_loader = create_loaders(hparams, train_synthetic_ds, validation_synthetic_ds, test_synthetic_ds)

    num_epochs = hparams["epochs"]
    optimizer = AdamW(network.parameters(), hparams["learning_rate"])
    criterion = long_tail_fine_tune_loss
    network.to(DEVICE)

    best_model_state = {}
    best_loss = torch.inf
    not_improved_epochs = 0
    best_loss_epoch = 0
    tr_losses = []
    tr_global_accuracies = []
    tr_real_accuracies = []
    tr_synthetic_accuracies = []
    tv_losses = []
    tv_global_accuracies = []
    tv_real_accuracies = []
    tv_synthetic_accuracies = []
    for epoch in range(1, num_epochs + 1):
        network.to(DEVICE)
        network.train()
        losses = []
        global_accs = []
        real_accs = []
        synthetic_accs = []

        synthetic_dataloader_iterator = iter(train_synthetic_loader)
        for batch_idx, (real_data, real_target) in enumerate(train_real_loader):
            try:
                (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
            except StopIteration:
                synthetic_dataloader_iterator = iter(train_synthetic_loader)
                (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
            real_data, real_target = real_data.to(DEVICE), real_target.to(DEVICE)
            synthetic_data, synthetic_target = synthetic_data.to(DEVICE), synthetic_target.to(DEVICE)
            real_pred = network(real_data)
            synthetic_pred = network(synthetic_data)
            optimizer.zero_grad()
            loss = criterion(real_pred, real_target, synthetic_pred, synthetic_target, hparams["weight_factor"])
            loss.backward()
            acc_real = 100 * (correct_predictions(real_pred, real_target) / real_data.shape[0])
            acc_synthetic = 100 * (correct_predictions(synthetic_pred, synthetic_target) / synthetic_data.shape[0])
            global_acc = 1/2*(acc_real+acc_synthetic)
            losses.append(loss.item())
            real_accs.append(acc_real)
            synthetic_accs.append(acc_synthetic)
            global_accs.append(global_acc)
            optimizer.step()
            if batch_idx % 5 == 0 or batch_idx >= len(train_real_loader):
                log.info(
                    f"Train Epoch: {epoch} [{batch_idx * len(real_data)}/{len(train_real_loader.dataset)} "
                    f"({100. * batch_idx / len(train_real_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAcc: {global_acc:.1f}"
                )

        tr_loss = np.mean(losses)
        tr_global_acc = np.mean(global_accs)
        tr_real_acc = np.mean(real_accs)
        tr_synthetic_acc = np.mean(synthetic_accs)

        network.eval()
        val_loss = 0
        val_global_acc = 0
        with torch.no_grad():
            synthetic_dataloader_iterator = iter(validation_synthetic_loader)
            for batch_idx, (real_data, real_target) in enumerate(validation_real_loader):
                try:
                    (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
                except StopIteration:
                    synthetic_dataloader_iterator = iter(validation_synthetic_loader)
                    (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
                real_data, real_target = real_data.to(DEVICE), real_target.to(DEVICE)
                synthetic_data, synthetic_target = synthetic_data.to(DEVICE), synthetic_target.to(DEVICE)
                real_pred = network(real_data)
                synthetic_pred = network(synthetic_data)
                batch_val_loss = criterion(real_pred, real_target, synthetic_pred, synthetic_target, weight_factor=hparams["weight_factor"])
                val_loss = val_loss + batch_val_loss.item()
                acc_real = 100 * (correct_predictions(real_pred, real_target) / real_data.shape[0])
                acc_synthetic = 100 * (correct_predictions(synthetic_pred, synthetic_target) / synthetic_data.shape[0])
                val_global_acc = 1/2*(acc_real+acc_synthetic)
        tv_loss = val_loss / len(validation_real_loader.dataset)
        tv_global_acc = 100 * val_global_acc / len(validation_real_loader.dataset)
        tv_real_acc = 100 * val_global_acc / len(validation_real_loader.dataset)
        tv_synthetic_acc = 100 * val_global_acc / len(validation_real_loader.dataset)
        log.info(
            f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_global_acc}/{len(validation_real_loader.dataset)} ({val_global_acc:.0f}%)"
        )
        tv_losses.append(tv_loss)
        tv_global_accuracies.append(tv_global_acc)
        tv_real_accuracies.append(tv_real_acc)
        tv_synthetic_accuracies.append(tv_synthetic_acc)
        tr_losses.append(tr_loss)
        tr_global_accuracies.append(tr_global_acc)
        tr_real_accuracies.append(tr_real_acc)
        tr_synthetic_accuracies.append(tr_synthetic_acc)

        if tv_loss <= (best_loss - early_stop.min_delta):
            best_loss = tv_loss
            best_loss_epoch = epoch
            best_model_state = network.state_dict()
            not_improved_epochs = 0
            log.warning(
                f"Best loss {best_loss} in epoch {best_loss_epoch} from total of {num_epochs + 1} epochs"
            )
        else:
            not_improved_epochs = not_improved_epochs + 1 if hparams["earlystop"] else 0
            if not_improved_epochs == early_stop.patience:
                log.warning(
                    f"{early_stop.patience} without improvements, STOP training \n"
                    f"Best loss {best_loss} in epoch {best_loss_epoch} from total of {num_epochs + 1} epochs"
                )
                break

    network.eval()
    raw_predictions = []
    targets = []
    with torch.no_grad():
        synthetic_dataloader_iterator = iter(test_synthetic_loader)
        for batch_idx, (real_data, real_target) in enumerate(test_real_loader):
            try:
                (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
            except StopIteration:
                synthetic_dataloader_iterator = iter(test_synthetic_loader)
                (synthetic_data, synthetic_target) = next(synthetic_dataloader_iterator)
            real_data, real_target = real_data.to(DEVICE), real_target.to(DEVICE)
            synthetic_data, synthetic_target = synthetic_data.to(DEVICE), synthetic_target.to(DEVICE)
            real_pred = network(real_data)
            synthetic_pred = network(synthetic_data)
            raw_predictions = raw_predictions + list(real_pred.cpu().detach().numpy()) + list(synthetic_pred.cpu().detach().numpy())
            targets = targets + list(real_target.cpu().detach().numpy()) + list(synthetic_target.cpu().detach().numpy())

    predictions_targets = (raw_predictions, targets)

    metrics = {
        "tr_losses": tr_losses,
        "tv_losses": tv_losses,
        "tr_accuracies": tr_global_accuracies,
        "tv_accuracies": tv_global_accuracies,
        "tr_real_accuracies": tr_real_accuracies,
        "tv_real_accuracies": tv_real_accuracies,
        "tr_synthetic_accuracies": tr_synthetic_accuracies,
        "tv_synthetic_accuracies": tv_synthetic_accuracies,
        "predictions_and_targets": predictions_targets,
    }
    if save_results:
        save_model_results(metrics, results_path / (exp_name + ".csv"))
    if save_model:
        torch.save(best_model_state, model_path / (exp_name + ".pt"))

    return metrics

def create_loaders(hparams, train_set, validation_set, test_set):
    train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True, drop_last=True)
    validation_loader = DataLoader(
        validation_set, batch_size=hparams["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_set, batch_size=hparams["batch_size"], shuffle=True, drop_last=True)
    return train_loader, validation_loader, test_loader
