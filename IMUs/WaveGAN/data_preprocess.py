"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar,
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data,"
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_preprocess(
        file_name: str,
        max_seq_len: int,
        padding_value: float = -1.0,
        impute_method: str = "mode",
        scaling_method: str = "minmax",
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Remove outliers
    2. Extract sequence length for each patient id
    3. Impute missing data
    4. Normalize data
    6. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - max_seq_len (int): maximum sequence length
    - impute_method (str): The imputation method ("median" or "mode")
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - processed_data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data
    """

    #########################
    # Load data
    #########################

    index = 'Idx'

    # Load csv
    print("Loading data...\n")
    ori_data = pd.read_csv(file_name)

    # Remove spurious column, so that column 0 is now 'admissionid'.
    if ori_data.columns[0] == "Unnamed: 0":
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

    # Extract labels and remove them from ori_data
    labels = ori_data[['Label', index]].groupby(index).first()['Label'].values # One label per index
    ori_data = ori_data.drop(['Label'], axis=1)



    #########################
    # Remove outliers from dataset
    #########################

    no = ori_data.shape[0]
    z_scores = stats.zscore(ori_data, axis=0, nan_policy='omit')
    z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3
    ori_data = ori_data[z_filter]

    # Parameters
    uniq_id = np.unique(ori_data[index])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 2

    #########################
    # Determine global min and max
    #########################
    scaling_data = ori_data.drop('Idx', axis=1)
    scaling_data = scaling_data.drop('0', axis=1)
    global_min = scaling_data.min().min()
    global_max = scaling_data.max().max()

    params = [global_min, global_max]


    #########################
    # Impute, scale and pad data
    #########################

    # Initialize scaler
    """if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(ori_data.iloc[:, 2:])
        #scaler.fit(ori_data)
        params = [scaler.data_min_, scaler.data_max_]

    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(ori_data)
        params = [scaler.mean_, scaler.var_]"""

    # Imputation values
    if impute_method == "median":
        impute_vals = ori_data.median()
    elif impute_method == "mode":
        impute_vals = stats.mode(ori_data).mode[0]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")

    # Output initialization
    output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    output.fill(padding_value)
    time = []
    grouped_labels = np.empty(no, dtype=labels.dtype)  # Initialize array for grouped labels

    # For each uniq id
    for i in tqdm(range(no)):
        # Extract the time-series data with a certain admissionid

        curr_data = ori_data[ori_data[index] == uniq_id[i]].to_numpy()

        # Impute missing data
        curr_data = imputer(curr_data, impute_vals)

        # Normalize data
        curr_data[:, 1:] = (curr_data[:, 1:] - global_min) / (global_max - global_min)
        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = curr_data.shape[0]

        # Pad data to `max_seq_len`
        if curr_no >= max_seq_len:
            output[i, :, :] = curr_data[:max_seq_len, 2:]  # Shape: [1, max_seq_len, dim] # Drop Idx column
            time.append(max_seq_len)
        else:
            output[i, :curr_no, :] = curr_data[:, 2:]  # Shape: [1, max_seq_len, dim] # Drop Idx column
            time.append(curr_no)

        # Store the label for the current group
        grouped_labels[i] = labels[i]

    return output, time, params, max_seq_len, padding_value, grouped_labels


def imputer(
        curr_data: np.ndarray,
        impute_vals: List,
        zero_fill: bool = True
) -> np.ndarray:
    """Impute missing data given values for each columns.

    Args:
        curr_data (np.ndarray): Data before imputation.
        impute_vals (list): Values to be filled for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where
            impute_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    impute_vals = pd.Series(impute_vals)

    # Impute data
    imputed_data = curr_data.fillna(impute_vals)

    # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
    imputed_data = imputed_data.fillna(0.0)

    # Check for any N/A values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()
