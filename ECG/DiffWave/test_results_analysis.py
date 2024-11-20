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

COLUMNS_OF_INTEREST = ["accuracy", "precision", "recall", "f1_score", "roc_auc_score"]
EXPERIMENTS_OF_INTEREST = [
    # ;experiment_name
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
if __name__ == "__main__":
    results_file = ARTIFACTS_PATH / "results/test_results.csv"
    plots_path = ARTIFACTS_PATH / "plots"
    df = pd.read_csv(results_file, sep=";", index_col="experiment_name")
    df = df.loc[EXPERIMENTS_OF_INTEREST, COLUMNS_OF_INTEREST]
    cols = []
    for exp in EXPERIMENTS_OF_INTEREST:
        if exp.endswith("PTBXLRhythm"):
            cols.append(("PTBXLRhythm", exp))
        else:
            cols.append(("CHAPMAN", exp))
    df.index = pd.MultiIndex.from_tuples(cols)
    latex_df = df.to_latex(bold_rows=True, label="tab:table_results_1")
    print(latex_df)
