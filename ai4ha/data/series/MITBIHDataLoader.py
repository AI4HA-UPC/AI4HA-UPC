# load mitbih dataset

from lib2to3.fixes.fix_isinstance import FixIsinstance
import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

cls_dit = {
    "Non-Ectopic Beats": 0,
    "Superventrical Ectopic": 1,
    "Ventricular Beats": 2,
    "Unknown": 3,
    "Fusion Beats": 4,
}


def last_samples(path):
    dirs = os.listdir(path)
    dirs = sorted([d for d in dirs if d.startswith("sampled_data")])
    return int(dirs[-1].split(".")[0].split("_")[-1])


class MITBIHtrain(Dataset):

    def __init__(self,
                 filename="./mitbih_train.csv",
                 n_samples=20000,
                 oneD=False,
                 resamp=True,
                 padding=0,
                 padalgo="zero",
                 normalize=False):
        data_train = pd.read_csv(filename, header=None)

        # making the class labels for our dataset
        data_0 = data_train[data_train[187] == 0]
        data_1 = data_train[data_train[187] == 1]
        data_2 = data_train[data_train[187] == 2]
        data_3 = data_train[data_train[187] == 3]
        data_4 = data_train[data_train[187] == 4]

        if resamp:
            data_0_resample = resample(data_0,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_1_resample = resample(data_1,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_2_resample = resample(data_2,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_3_resample = resample(data_3,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_4_resample = resample(data_4,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)

            train_dataset = pd.concat((
                data_0_resample,
                data_1_resample,
                data_2_resample,
                data_3_resample,
                data_4_resample,
            ))
        else:
            train_dataset = pd.concat((data_0, data_1, data_2, data_3, data_4))
            for i, d in enumerate((data_0, data_1, data_2, data_3, data_4)):
                print(f"Class {i} = {d.shape}")

        self.X_train = train_dataset.iloc[:, :-1].values
        # scaling to -1 to 1 range
        if normalize:
            vmax = np.max(self.X_train)
            vmin = np.min(self.X_train)
            self.X_train = (self.X_train - vmin) / (vmax - vmin)

        # reshaping the data to fit the model (1, series length)
        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1,
                                                self.X_train.shape[1])
            length = self.X_train.shape[2]
            if padding != 0:
                if padalgo == 'zero':
                    tmp = np.zeros(
                        (self.X_train.shape[0], 1, length + padding))
                    tmp[:, :, :self.X_train.shape[2]] = self.X_train
                    self.X_train = tmp
                elif padalgo == 'repeat':
                    tmp = np.zeros(
                        (self.X_train.shape[0], 1, length + padding))
                    tmp[:, :, :self.X_train.shape[2]] = self.X_train
                    for i in range(length, length + padding):
                        tmp[:, :, i] = self.X_train[:, :,
                                                    self.X_train.shape[2] - 1]
                    self.X_train = tmp
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1,
                                                self.X_train.shape[1])
            length = self.X_train.shape[3]
            if padding != 0:
                if padalgo == 'zero':
                    tmp = np.zeros(
                        (self.X_train.shape[0], 1, 1, length + padding))
                    tmp[:, :, :, :self.X_train.shape[3]] = self.X_train
                    self.X_train = tmp
                elif padalgo == 'repeat':
                    tmp = np.zeros(
                        (self.X_train.shape[0], 1, 1, length + padding))
                    tmp[:, :, :, :self.X_train.shape[3]] = self.X_train
                    for i in range(length, length + padding):
                        tmp[:, :, :,
                            i] = self.X_train[:, :, :,
                                              self.X_train.shape[3] - 1]
                    self.X_train = tmp
        self.y_train = train_dataset[187].values

        print(f"X_train shape is {self.X_train.shape}")
        print(f"y_train shape is {self.y_train.shape}")
        print(f'PAD={padding} PADALGO={padalgo}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class MITBIHtest(Dataset):

    def __init__(self,
                 filename="./mitbih_test.csv",
                 n_samples=1000,
                 oneD=False,
                 resamp=True,
                 fixsize=None,
                 normalize=False):
        print("--------------------------------")
        print("-------TEST DATA-------")
        data_test = pd.read_csv(filename, header=None)

        # making the class labels for our dataset
        data_0 = data_test[data_test[187] == 0]
        data_1 = data_test[data_test[187] == 1]
        data_2 = data_test[data_test[187] == 2]
        data_3 = data_test[data_test[187] == 3]
        data_4 = data_test[data_test[187] == 4]

        if resamp:
            data_0_resample = resample(data_0,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_1_resample = resample(data_1,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_2_resample = resample(data_2,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_3_resample = resample(data_3,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)
            data_4_resample = resample(data_4,
                                       n_samples=n_samples,
                                       random_state=123,
                                       replace=True)

            test_dataset = pd.concat((
                data_0_resample,
                data_1_resample,
                data_2_resample,
                data_3_resample,
                data_4_resample,
            ))
        else:
            test_dataset = pd.concat((data_0, data_1, data_2, data_3, data_4))
            for i, d in enumerate((data_0, data_1, data_2, data_3, data_4)):
                print(f"Class {i} = {d.shape}")

        self.X_test = test_dataset.iloc[:, :-1].values
        # scaling to -1 to 1 range
        if normalize:
            vmax = np.max(self.X_train)
            vmin = np.min(self.X_train)
            self.X_train = (self.X_train - vmin) / (vmax - vmin)

        # reshaping the data to fit the model (1, series length)
        if oneD:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1,
                                              self.X_test.shape[1])
            if fixsize is not None:
                tmp = np.zeros((self.X_test.shape[0], 1, fixsize))
                tmp[:, :, :self.X_test.shape[2]] = self.X_test
                self.X_test = tmp
        else:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 1,
                                              self.X_test.shape[1])
            if fixsize is not None:
                tmp = np.zeros((self.X_test.shape[0], 1, 1, fixsize))
                tmp[:, :, :, :self.X_test.shape[3]] = self.X_test
                self.X_test = tmp
        self.y_test = test_dataset[187].values

        print(f"X_test shape is {self.X_test.shape}")
        print(f"y_test shape is {self.y_test.shape}")
        # print(f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]


class MITBIHgenerated(Dataset):
    def __init__(self, PATH="", DIR="", SAMP=0, var=1):
        SAMP = last_samples(f"{PATH}/{DIR}/Samples")
        ldata = np.load(f"{PATH}/{DIR}/Samples/sampled_data_v{var}_{SAMP:05d}.npz")
        self.X_test = ldata["samples"]
        self.y_test = ldata["classes"]

        print(f"X_test shape is {self.X_test.shape}")
        print(f"y_test shape is {self.y_test.shape}")

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]


class MITBIHmix(Dataset):
    def __init__(
        self,
        filename="./mitbih_train.csv",
        n_samples=1000,
        oneD=False,
        resamp=True,
        PATH="",
        DIR="",
        SAMP=0,
        classes=None,
        var=1,
    ):
        print("--------------------------------")
        print("-------MIXED DATA-------")
        data_test = pd.read_csv(filename, header=None)

        # making the class labels for our dataset
        data_0 = data_test[data_test[187] == 0]
        data_1 = data_test[data_test[187] == 1]
        data_2 = data_test[data_test[187] == 2]
        data_3 = data_test[data_test[187] == 3]
        data_4 = data_test[data_test[187] == 4]

        if resamp:
            data_0_resample = resample(
                data_0, n_samples=n_samples, random_state=123, replace=True
            )
            data_1_resample = resample(
                data_1, n_samples=n_samples, random_state=123, replace=True
            )
            data_2_resample = resample(
                data_2, n_samples=n_samples, random_state=123, replace=True
            )
            data_3_resample = resample(
                data_3, n_samples=n_samples, random_state=123, replace=True
            )
            data_4_resample = resample(
                data_4, n_samples=n_samples, random_state=123, replace=True
            )

            test_dataset = pd.concat(
                (
                    data_0_resample,
                    data_1_resample,
                    data_2_resample,
                    data_3_resample,
                    data_4_resample,
                )
            )
        else:
            test_dataset = pd.concat((data_0, data_1, data_2, data_3, data_4))
            for i, d in enumerate((data_0, data_1, data_2, data_3, data_4)):
                print(f"Class {i} = {d.shape}")

        self.X_test = test_dataset.iloc[:, :-1].values
        if oneD:
            self.X_test = self.X_test.reshape(
                self.X_test.shape[0], 1, self.X_test.shape[1]
            )
        else:
            self.X_test = self.X_test.reshape(
                self.X_test.shape[0], 1, 1, self.X_test.shape[1]
            )
        self.y_test = test_dataset[187].values

        print(f"X_test shape is {self.X_test.shape}")
        print(f"y_test shape is {self.y_test.shape}")

        SAMP = last_samples(f"{PATH}/{DIR}/Samples")
        ldata = np.load(f"{PATH}/{DIR}/Samples/sampled_data_v{var}_{SAMP:05d}.npz")
        X_test_g = ldata["samples"]
        y_test_g = ldata["classes"]

        if classes is not None:
            y_test_sel = [y_test_g == classes[0]]

            for c in classes[1:]:
                y_test_sel = np.logical_or(y_test_sel, y_test_g == c)

            self.X_test = np.concatenate(
                (self.X_test, X_test_g[np.squeeze(y_test_sel)])
            )
            self.y_test = np.concatenate(
                (self.y_test, y_test_g[np.squeeze(y_test_sel)])
            )
        else:
            self.X_test = np.concatenate(self.X_test, X_test_g)
            self.y_test = np.concatenate(self.y_test, y_test_g)

        print(f"X_test shape is {self.X_test.shape}")
        print(f"y_test shape is {self.y_test.shape}")

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]
