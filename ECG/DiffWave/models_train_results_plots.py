from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
import matplotlib as mpl

EXPERIMENTS_TO_LOAD = "ecg_hparams_tf_ft.yml"
with open("configurations/" + EXPERIMENTS_TO_LOAD) as fp:
    EXPERIMENTS_TF_FT = yaml.safe_load(fp)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)
COLUMNS_OF_INTEREST = ["tv_losses"]


if __name__ == "__main__":
    experiments = EXPERIMENTS_TF_FT["experiments"]
    results_path = ARTIFACTS_PATH / "results"
    plots_path = ARTIFACTS_PATH / "plots"

    df_plot = pd.DataFrame()
    for experiment_name in experiments:
        file_path = results_path.glob(f"*{experiment_name}*")
        for file in file_path:
            df = pd.read_csv(file, index_col=0)
            df_plot = pd.concat((df_plot, df[COLUMNS_OF_INTEREST]), axis=1)
            parts = file.stem.split("_")
            print(file.stem)
            col_name = "_".join([parts[1], parts[3], parts[6], parts[7]])
            df_plot.rename({COLUMNS_OF_INTEREST[-1]: col_name}, axis=1, inplace=True)

    c = np.arange(1, len(df_plot.columns) * 2 + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap1 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
    cmap1.set_array([])
    cmap2.set_array([])
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    for i, col_name in enumerate(df_plot):
        if "synthetic" in col_name:
            line_color = cmap1.to_rgba(i + 11)
            line_style = "solid"
        else:
            line_color = cmap2.to_rgba(i + 11)
            line_style = "dashed"
        ax.plot(df_plot[col_name], label=col_name, c=line_color, linestyle=line_style)
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 1.2])
    ax.set_title("Validation loss during train")
    plt.legend()
    plt.savefig(plots_path / "models_validations_loss.png")
    plt.close()
