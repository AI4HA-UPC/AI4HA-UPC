import csv
from pathlib import Path


def save_experiments_results(
    results_file_path: Path,
    experiment_name: str,
    scores: dict,
    experiment_configs: dict,
):
    row = {"experiment_name": experiment_name, **scores, **experiment_configs}
    keep_keys = [
        "experiment_name",
        "dataset_size",
        "data_split",
        "execution_time",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc_score",
        "dataset",
        "model_hparams",
        "generative_model",
        "classifier",
    ]
    row = {k: row.get(k) for k in keep_keys}
    if not results_file_path.exists():
        with open(results_file_path, "a") as fp:
            column_index = list(row.keys())
            writer = csv.DictWriter(fp, fieldnames=column_index, delimiter=";")
            writer.writeheader()
    with open(results_file_path, "a") as fp:
        column_index = list(row.keys())
        writer = csv.DictWriter(fp, fieldnames=column_index, delimiter=";")
        writer.writerow(row)
