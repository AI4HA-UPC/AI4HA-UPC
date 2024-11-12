"""
This code was extracted from https://github.com/jsyoon0823/TimeGAN/blob/master/utils.py
with the pertinent modifications to run on pytorch instead of tensorflow
"""
import torch
import torch.nn as nn
import numpy as np


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(data.shape[1]):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))

    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.
    Args:
      - module_name: gru, lstm, or lstmLN

    Returns:
      - rnn_cell: RNN Cell
    """
    assert module_name in ['gru', 'lstm', 'lstmLN']

    # GRU
    if module_name == 'gru':
        rnn_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
    # LSTM
    elif module_name == 'lstm':
        rnn_cell = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
    # LSTM with Layer Normalization
    elif module_name == 'lstmLN':
        # Note: PyTorch does not have a built-in LayerNorm LSTMCell.
        raise NotImplementedError(
            "Layer normalized LSTM cell is not natively available in PyTorch.")

    return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim: dimension of random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    print(data)
    no = len(data)
    print(f'no:{no}')
    idx = np.random.permutation(no)
    print(f'permutation: {idx}')
    print(f'Batch:{batch_size}, {idx[:batch_size]}')
    train_idx = idx[:batch_size]

    print(data[:, 0])
    X_mb = list(data[:, i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def NormMinMax(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val  # [3661, 24, 6]

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val

