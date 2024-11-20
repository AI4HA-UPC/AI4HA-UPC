from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from datasets.loaders_ecg import DATASET_SIZES

ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)

EXPERIMENTS_OF_INTEREST = [
    "experiment_real_PTBXL_1dcnn",
    "experiment_real_CHAPMAN_1dcnn",
    "experiment_PTBXL_1dcnn_time_vqvae_TrRSTeR",
    "experiment_CHAPMAN_1dcnn_time_vqvae_TrRSTeR",
    "experiment_PTBXL_transformer_time_vqvae_TrRSTeR",
    "experiment_CHAPMAN_transformer_time_vqvae_TrRSTeR",
    "experiment_PTBXL_1dcnn_all_synthetic_TrRSTeR",
    "experiment_CHAPMAN_1dcnn_all_synthetic_TrRSTeR",
    "experiment_PTBXL_transformer_all_synthetic_TrRSTeR",
    "experiment_CHAPMAN_transformer_all_synthetic_TrRSTeR",
    "experiment_ptbxl_finetune_1dcnn_size_small_PTBXLRhythm",
    "experiment_ptbxl_finetune_1dcnn_size_medium_PTBXLRhythm",
    "experiment_ptbxl_finetune_1dcnn_size_large_PTBXLRhythm",
    "experiment_ptbxl_finetune_1dcnn_size_extra_PTBXLRhythm",
    "experiment_ptbxl_finetune_1dcnn_size_full_PTBXLRhythm",
    "experiment_chapman_finetune_1dcnn_size_small_CHAPMAN",
    "experiment_chapman_finetune_1dcnn_size_medium_CHAPMAN",
    "experiment_chapman_finetune_1dcnn_size_large_CHAPMAN",
    "experiment_chapman_finetune_1dcnn_size_extra_CHAPMAN",
    "experiment_chapman_finetune_1dcnn_size_full_CHAPMAN",
]
COMPARED_EXPERIMENTS = [
    ("experiment_ptbxl_finetune_1dcnn_size_", "experiment_real_PTBXL_1dcnn"),
]

RESULTS_FILE_COLUMNS = [
    "experiment_name",
    "dataset_size",
    "execution_time",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc_score",
    "dataset",
    "generative_model",
    "classifier",
]
COLUMNS_OF_INTEREST = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc_score",
    "execution_time",
]

if __name__ == "__main__":
    results_file = ARTIFACTS_PATH / "results/results.csv"
    tf_results_file = ARTIFACTS_PATH / "results/tf_results.csv"
    plots_path = ARTIFACTS_PATH / "plots"
    df_results = pd.read_csv(
        results_file, sep=";", index_col="experiment_name", usecols=RESULTS_FILE_COLUMNS
    )
    df_tf_results = pd.read_csv(
        tf_results_file,
        sep=";",
        index_col="experiment_name",
        usecols=RESULTS_FILE_COLUMNS,
    )
    df = pd.concat([df_results, df_tf_results])
    df = df.loc[EXPERIMENTS_OF_INTEREST]
    df = df.sort_values("dataset_size")
    sizes = DATASET_SIZES
    table_columns = []
    for c in COLUMNS_OF_INTEREST:
        table_columns.append(f"{c}_synthetic")
        table_columns.append(f"{c}_real")
    tuples = [(c.split("_")[0], c.split("_")[-1]) for c in table_columns]
    multi_index = pd.MultiIndex.from_tuples(tuples)
    for exp1, exp2 in COMPARED_EXPERIMENTS:
        df1 = df[df.index.str.startswith(exp1)]
        df1.index = sizes
        df2 = df[df.index.str.startswith(exp2)]
        df2.index = sizes  # TODO: train real with different % of data in order to compare scores...
        df_compare = df1.join(df2, lsuffix="_synthetic", rsuffix="_real")
        df_compare = df_compare[table_columns]
        df_compare.columns = multi_index
        df_compare.reset_index(names="percentage", inplace=True)
        print(
            df_compare.to_latex(index=False, bold_rows=True, label="tab:table_results")
        )

        fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
        axs = axs.ravel()
        for ax, col in zip(axs, COLUMNS_OF_INTEREST):
            print(exp1, exp2)
            print((df1[col], df2[col]))
            ax.plot(df1[col], label="synthetic", c="r")
            ax.plot(df2[col], label="real", c="black")
            ax.set_title(col)
        handles, labels = ax.get_legend_handles_labels()
        fig.suptitle(f"{exp1} vs {exp2}")
        fig.legend(handles, labels, loc="center right")
        plt.savefig(plots_path / f"FT_results_model_{COMPARED_EXPERIMENTS}.png")
        plt.close()
