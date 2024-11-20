from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("max_colwidth", -1)
ARTIFACTS_PATH = (
    Path("../artifacts/ecg")
    if Path("../artifacts").exists()
    else Path("/home/bsc70/bsc70582/artifacts/ecg")
)

COLUMNS_OF_INTEREST = ["accuracy", "precision", "recall"]
EXPERIMENTS_OF_INTEREST = [
    "train_real_synthetic_test_real_MIT",
    "train_synthetic_test_real_MIT",
    "train_synthetic_test_synthetic_MIT",
    "train_real_test_real_MIT",
    "train_synthetic_test_real_PTB",
    "train_synthetic_test_synthetic_PTB",
    "train_real_test_real_PTB",
]
if __name__ == "__main__":
    results_file = ARTIFACTS_PATH / "results/test_results.csv"
    plots_path = ARTIFACTS_PATH / "plots"
    df = pd.read_csv(results_file, sep=";", index_col="experiment_name")
    df = df.loc[EXPERIMENTS_OF_INTEREST, COLUMNS_OF_INTEREST]
    cols = []
    for exp in EXPERIMENTS_OF_INTEREST:
        if exp.endswith("MIT"):
            cols.append(("MIT", exp))
        else:
            cols.append(("PTB", exp))
    df.index = pd.MultiIndex.from_tuples(cols)
    latex_df = df.to_latex(bold_rows=True, label="tab:table_results_1")
    print(latex_df)
