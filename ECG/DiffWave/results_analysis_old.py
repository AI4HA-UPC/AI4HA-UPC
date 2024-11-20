from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from datasets.loaders_ecg import DATASET_SIZES

ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)

COLUMNS_OF_INTEREST = ["accuracy", "precision", "recall", "execution_time"]
EXPERIMENTS_OF_INTEREST = [
    "experiment_synthetic_MIT_FT_size_small_KaggleMIT",
    "experiment_synthetic_MIT_FT_size_medium_KaggleMIT",
    "experiment_synthetic_MIT_FT_size_large_KaggleMIT",
    "experiment_synthetic_MIT_FT_size_extra_KaggleMIT",
    "experiment_synthetic_MIT_FT_size_full_KaggleMIT",
    "experiment_real_MIT_size_small_KaggleMIT",
    "experiment_real_MIT_size_medium_KaggleMIT",
    "experiment_real_MIT_size_large_KaggleMIT",
    "experiment_real_MIT_size_extra_KaggleMIT",
    "experiment_real_MIT_size_full_KaggleMIT",
    "experiment_synthetic_PTB_size_small_KagglePTB",
    "experiment_synthetic_PTB_size_medium_KagglePTB",
    "experiment_synthetic_PTB_size_large_KagglePTB",
    "experiment_synthetic_PTB_size_extra_KagglePTB",
    "experiment_synthetic_PTB_size_full_KagglePTB",
    "experiment_real_PTB_size_small_KagglePTB",
    "experiment_real_PTB_size_medium_KagglePTB",
    "experiment_real_PTB_size_large_KagglePTB",
    "experiment_real_PTB_size_extra_KagglePTB",
    "experiment_real_PTB_size_full_KagglePTB",
    "experiment_synthetic_MIT_real_PTB_size_small_KagglePTB",
    "experiment_synthetic_MIT_real_PTB_size_medium_KagglePTB",
    "experiment_synthetic_MIT_real_PTB_size_large_KagglePTB",
    "experiment_synthetic_MIT_real_PTB_size_extra_KagglePTB",
    "experiment_synthetic_MIT_real_PTB_size_full_KagglePTB",
]
COMPARED_EXPERIMENTS = [
    ("experiment_synthetic_MIT_FT", "experiment_real_MIT"),
    ("experiment_synthetic_PTB", "experiment_real_PTB"),
    ("experiment_synthetic_MIT_real_PTB", "experiment_real_PTB"),
]
if __name__ == "__main__":
    results_file = ARTIFACTS_PATH / "results/results.csv"
    plots_path = ARTIFACTS_PATH / "plots"
    df = pd.read_csv(results_file, sep=";", index_col="experiment_name")
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
        df2.index = sizes
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
