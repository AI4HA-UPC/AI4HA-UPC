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


def resample_segment(dataframe, resample_len=64):
    	if dataframe.empty:
    		return np.zeros((resample_len, len(dataframe.columns)), dtype=np.float32)
    		
    	resampled_data = np.zeros((resample_len, len(dataframe.columns)), dtype=np.float32)
    	for idx, column in enumerate(dataframe.columns):
    		y = dataframe[column].values
    		if len(y) == 0:
    			resampled_data[:, idx] = 0
    			continue
    		x_original = np.linspace(0, 1, len(y))
    		x_resampled = np.linspace(0, 1, resample_len)
    		resampled_data[:, idx] = np.interp(x_resampled, x_original, y)
    		
    	return resampled_data
    	
def data_preprocess(
	df: pd.DataFrame, 
        max_seq_len: int,
        resample_len: int = 64,
        padding_value: float = 0,
        impute_method: str = "mode",
        scaling_params: Tuple[float, float] = None,
) -> Tuple[np.ndarray, int]:
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
    # Remove outliers from dataset
    #########################
    
    z_scores = stats.zscore(df, axis=0, nan_policy='omit')
    z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3
    df = df[z_filter]

    #########################
    # Impute, scale and pad data
    #########################

    # Imputation values
    if impute_method == "median":
        impute_vals = df.median()
    elif impute_method == "mode":
        impute_vals = stats.mode(df, nan_policy='omit').mode[0]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")
        
    df_imputed = pd.DataFrame(imputer(df.to_numpy(), impute_vals))
    
    # Scaling
    if scaling_params:
    	global_min, global_max = scaling_params
    	df_scaled = (df_imputed - global_min) / (global_max - global_min)
    else:
    	global_min = df_imputed.min().min()
    	global_max = df_imputed.max().max()
    	df_scaled = (df_imputed - global_min) / (global_max - global_min)
  	
    # Convert to numpy array
    data_array = df_scaled.to_numpy()

    # Resample data to fixed length
    resampled_data = resample_segment(df_scaled, resample_len=resample_len)
    
    # Sequence length
    seq_len = resampled_data
    
    
    return resampled_data, seq_len
    

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
