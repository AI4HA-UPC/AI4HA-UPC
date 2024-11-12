# -*- coding: UTF-8 -*-
import numpy as np
import torch


class TimeGANDataset(torch.utils.data.Dataset):
    """TimeGAN Dataset for sampling data with their respective time

    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """

    def __init__(self, data, time=None, labels=None, padding_value=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(f"len(data) `{len(data)}` != len(time) {len(time)}")
        if len(data) != len(labels):
            raise ValueError(f"len(data) `{len(data)}` != len(labels) {len(labels)}")

        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)
        self.Y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        current_data = self.X[idx]
        current_time = self.T[idx]
        current_label = self.Y[idx]
        
        # Conditionally set prev_data based on label match
        if idx > 0 and self.Y[idx - 1] == current_label:
            prev_data = self.X[idx - 1]
        else:
            prev_data = current_data

        return current_data, prev_data, current_time, current_label

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Unzip the batch
        X_mb, prev_X_mb, T_mb, Y_mb = zip(*batch)

        # Pad sequences to max length if necessary
        return torch.stack(X_mb), torch.stack(prev_X_mb), torch.stack(T_mb), torch.stack(Y_mb)

