import pathlib

import numpy as np
from pandas import crosstab
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from utils.plots import plot_confusion_matrix

from typing import NamedTuple


class Scores(NamedTuple):
    precision: float
    recall: float
    f1_score: float


def compute_classification_metrics(crosstable: np.ndarray, mode: str = "micro"):
    support = np.sum(crosstable, axis=1)
    num_classes = crosstable.shape[-1]
    acc = round(np.trace(crosstable) / crosstable.sum(), 3)
    precision = None
    recall = None
    f1_score = None
    cum_tp_fp = 0
    cum_tp_fn = 0
    cum_tp = 0
    label_scores = {}
    for label in range(num_classes):
        tp = crosstable[label, label]
        fp = np.sum(crosstable[label]) - tp
        fn = np.sum(crosstable[:, label]) - tp
        cum_tp = cum_tp + tp
        cum_tp_fp = cum_tp_fp + tp + fp
        cum_tp_fn = cum_tp_fn + tp + fn
        precision = round(tp / (tp + fp), 3)
        recall = round(tp / (tp + fn), 3)
        f1_score = round(2 * (precision * recall) / (precision + recall), 3)
        if np.isnan(precision):
            precision = 0
        if np.isnan(recall):
            recall = 0
        if np.isnan(f1_score):
            f1_score = 0
        label_scores[label] = Scores(precision, recall, f1_score)

    if mode == "macro":
        f1_score, precision, recall = compute_scores_macro(label_scores, num_classes)
    elif mode == "micro":
        precision = round(cum_tp / cum_tp_fp, 3)
        recall = round(cum_tp / cum_tp_fn, 3)
        f1_score = round(2 * (precision * recall) / (precision + recall), 3)
    return (
        round(acc, 3),
        round(precision, 3),
        round(recall, 3),
        round(f1_score, 3),
        support,
    )


def compute_scores_macro(label_scores, num_clases):
    precision = 0
    recall = 0
    f1_score = 0
    for score in label_scores.values():
        precision = precision + score.precision
        recall = recall + score.recall
        f1_score = f1_score + score.f1_score
    precision = precision / num_clases
    recall = recall / num_clases
    f1_score = f1_score / num_clases
    return f1_score, precision, recall


def contingency_matrix(
    preds: np.ndarray, targets: np.ndarray, files_path: pathlib.Path, name: str
):
    np_table, table, table_normalized = build_cross_table(preds, targets)
    acc, prec, rec, f1, _ = compute_classification_metrics(np_table, mode="macro")
    table_normalized.index.name = "predicted"
    table_normalized.columns.name = "true"
    plot_confusion_matrix(table, table_normalized, acc, prec, rec, f1, files_path, name)


def build_cross_table(preds, targets):
    total_instances = preds.shape[-1]
    table = crosstab(preds, targets, dropna=False)
    num_classes = len(np.unique(targets))
    np_table = np.zeros((num_classes, num_classes))
    for num_row, row in zip(table.index, table.values):
        np_table[num_row] = row
    table_normalized = table / total_instances
    return np_table, table, table_normalized


def compute_roc_auc(raw_preds, labels) -> float:
    targets = OneHotEncoder().fit_transform(np.array(labels).reshape(-1, 1))
    targets = np.array(targets.todense())
    return roc_auc_score(targets, raw_preds, average="macro")
