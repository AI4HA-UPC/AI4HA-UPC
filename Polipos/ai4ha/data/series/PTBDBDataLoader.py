# load PTBDB dataset

import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats': 0, 'Superventrical Ectopic': 1, 'Ventricular Beats': 2,
           'Unknown': 3, 'Fusion Beats': 4}


def last_samples(path):
    dirs = os.listdir(path)
    dirs = sorted([d for d in dirs if d.startswith("sampled_data")])
    return int(dirs[-1].split('.')[0].split('_')[-1])


class PTBDBtrain(Dataset):
    def __init__(self, filename=['./ptbdb_normal.csv', './ptbdb_abnormal.csv'], oneD=False, random_state=0):

        if len(filename) == 2:
            data_normal = pd.read_csv(filename[0], header=None)
            data_abnormal = pd.read_csv(filename[1], header=None)

            # making the class labels for our dataset
            data_normal = data_normal[data_normal[187] == 0]
            data_abnormal = data_abnormal[data_abnormal[187] == 1]

            train_dataset = pd.concat((data_normal, data_abnormal))
        else:
            train_dataset = pd.read_csv(filename[0], header=None)

        # for i, d in enumerate((data_normal, data_abnormal)):
        #         print(f"Class {i} = {d.shape}")

        X = train_dataset.iloc[:, :-1].values
        y = train_dataset[187].values

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=random_state, shuffle=True)

        if oneD:
            self.X_train = X_train.reshape(
                X_train.shape[0], 1, X_train.shape[1])
        else:
            self.X_train = X_train.reshape(
                X_train.shape[0], 1, 1, X_train.shape[1])
        self.y_train = y_train

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class PTBDBtest(Dataset):
    def __init__(self, filename=['./ptbdb_normal.csv', './ptbdb_abnormal.csv'], oneD=False, random_state=0):

        if len(filename) == 2:
            data_normal = pd.read_csv(filename[0], header=None)
            data_abnormal = pd.read_csv(filename[1], header=None)

            # making the class labels for our dataset
            data_normal = data_normal[data_normal[187] == 0]
            data_abnormal = data_abnormal[data_abnormal[187] == 1]

            train_dataset = pd.concat((data_normal, data_abnormal))
            # for i, d in enumerate((data_normal, data_abnormal)):
            #         print(f"Class {i} = {d.shape}")
        else:
            train_dataset = pd.read_csv(filename[0], header=None)

        X = train_dataset.iloc[:, :-1].values
        y = train_dataset[187].values

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state, shuffle=True)

        if oneD:
            self.X_train = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        else:
            self.X_train = X_test.reshape(
                X_test.shape[0], 1, 1, X_test.shape[1])
        self.y_train = y_test

        print(f'X_test shape is {self.X_train.shape}')
        print(f'y_test shape is {self.y_train.shape}')
        # print(f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class PTBDBgenerated(Dataset):
    def __init__(self, PATH='', DIR='', SAMP=0, var=1):
        SAMP = last_samples(f'{PATH}/{DIR}/Samples')
        ldata = np.load(
            f'{PATH}/{DIR}/Samples/sampled_data_v{var}_{SAMP:05d}.npz')
        self.X_test = ldata['samples']
        self.y_test = ldata['classes']

        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]


class PTBDBmix(Dataset):
    def __init__(self, filename=['./ptbdb_normal.csv', './ptbdb_abnormal.csv'],
                 oneD=False, random_state=0,
                 PATH='', DIR='', SAMP=0, var=1):

        if len(filename) == 2:
            data_normal = pd.read_csv(filename[0], header=None)
            data_abnormal = pd.read_csv(filename[1], header=None)

            # making the class labels for our dataset
            data_normal = data_normal[data_normal[187] == 0]
            data_abnormal = data_abnormal[data_abnormal[187] == 1]

            train_dataset = pd.concat((data_normal, data_abnormal))
            # for i, d in enumerate((data_normal, data_abnormal)):
            #         print(f"Class {i} = {d.shape}")
        else:
            train_dataset = pd.read_csv(filename[0], header=None)

        X = train_dataset.iloc[:, :-1].values
        y = train_dataset[187].values

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=random_state, shuffle=True)

        SAMP = last_samples(f'{PATH}/{DIR}/Samples')
        ldata = np.load(
            f'{PATH}/{DIR}/Samples/sampled_data_v{var}_{SAMP:05d}.npz')
        X_test_g = ldata['samples']
        y_test_g = ldata['classes']

        if oneD:
            self.X_train = np.concatenate(
                (X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), X_test_g))
        else:
            self.X_train = np.concatenate(
                (X_train.reshape(X_train.shape[0], 1, 1, X_train.shape[1]), X_test_g))
        self.y_train = np.concatenate((y_train, y_test_g))

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
