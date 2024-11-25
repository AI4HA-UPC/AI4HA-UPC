import os
import numpy as np
import pandas as pd
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from glob import glob
import re
warnings.filterwarnings("ignore")

ptbxl_class_dict = {'SBRAD': 0, 'SR': 1, 'AFIB': 2, 'STACH': 3,
        'AFLT': 4, 'SARRH': 5, 'SVTAC': 6}

class ptbxl_train(Dataset):
    def __init__(self, filename='', length=0):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:,-1].values
        self.X_train = data_train.iloc[:,:-1].values
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, length)
        # Shape = (N)umber samples  x (C)hannel=1 x L(ength sequence)
        
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class ptbxl_clusters(Dataset):
    def __init__(self, filename='', length=0):
        data_train = pd.read_csv(filename, header=None)
        self.labels = data_train.iloc[:,length].values
        self.X_train = data_train.iloc[:,:length].values
        self.clusters = data_train.iloc[:,length+1:].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X_train[idx], self.labels[idx], self.clusters[idx]

mitbih_class_dict = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1,
               'Ventricular Beats':2, 'Unknown':3, 'Fusion Beats':4}

class mitbih_train(Dataset):
    def __init__(self, filename='../../data/mitbih.csv', channels=1):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:,-1].values
        self.X_train = data_train.iloc[:, :-1].values
        length = (self.X_train.shape[1])//channels
        if (self.X_train.shape[1]/channels) > length:
            raise NameError("Wrong number of channels or uneven signal length")
        self.X_train.reshape(self.X_train.shape[0], channels, length)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

sleep_class_dict = {'Sleep stage W': 0, 'Sleep stage 1': 1,
              'Sleep stage 2': 2, 'Sleep stage 3/4': 3,
              'Sleep stage R': 4}

class sleep_train(Dataset):
    def __init__(self, filename='../data/sleep/isruc_F3-F4.csv', channels=1):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:,-1].values
        if channels == 1:
            self.X_train = data_train.iloc[:,:3000].values
        else:
            self.X_train = data_train.iloc[:,:-1].values
        length = (self.X_train.shape[1])//channels
        if (self.X_train.shape[1]/channels) > length:
            raise NameError("Wrong number of channels or uneven signal length")

        self.X_train = self.X_train.reshape(self.X_train.shape[0], channels, length)
        # Shape = (N)umber samples  x (C)hannels x L(ength sequence)
        
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class blanca_train(Dataset):
    def __init__(self, filename='../data/imu/concatenated_labels.csv', channels=6):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:,-1].values
        self.X_train = data_train.iloc[:,:-1].values
        length = (self.X_train.shape[1])//channels 
        if (self.X_train.shape[1]/channels) > length:
            raise NameError("Wrong number of channels or uneven signal length")

        self.X_train = self.X_train.reshape(self.X_train.shape[0], channels, length)
    
    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class multi_channel(Dataset):
    def __init__(self, filename='../data/ecg/ptbxl_100-combined.csv', channels=12):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:,-1].values
        self.X_train = data_train.iloc[:,:-1].values
        seq_length = (self.X_train.shape[1])//channels
        if (self.X_train.shape[1]/channels) > seq_length:
            raise NameError("Wrong number of channels or uneven signal length")
        self.X_train = self.X_train.reshape(self.X_train.shape[0], channels, seq_length)
        # Shape = (N)umber samples  x (C)hannels x (L)ength sequence

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class sleep_bitbrain(Dataset):
    def __init__(self, path):
        subjects = glob(path + '*.npz')
        dataset = []
        labels = []
        for s in subjects:
            data = np.load(s, allow_pickle=True)
            dataset.append(np.transpose(data['x'], (0, 2, 1)))
            labels.append(data['y'])
        self.X, self.Y = np.concatenate(dataset), np.concatenate(labels)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class sleep_NHP(Dataset):
    def __init__(self, path):
        subjects = sorted(glob(path + '*.npz'))
        dataset, labels = [], []
        for ids, s in enumerate(subjects):
            if ids >= 5:
                break
            data = np.load(s, allow_pickle=True)
            signal = np.transpose(data['x'], (0, 2, 1))
            dataset.append(signal[:,:1,:])
            labels.append(np.concatenate(data['y']))
            del data

        self.X, self.Y = np.concatenate(dataset), np.concatenate(labels)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
