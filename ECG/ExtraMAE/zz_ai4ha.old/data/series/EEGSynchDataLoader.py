# load mitbih dataset

import os 
import numpy as np
import pandas as pd
import sys 
from tqdm import tqdm 
import json
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def convert_string_data_to_values(value_string):
    str_list = json.loads(value_string)
    return str_list

QUALITY_THRESHOLD = 128


def preprocess(filename="/home/bejar/ssdstorage/eeg-data.csv"):
    eeg = pd.read_csv(filename)
    eeg = eeg.loc[eeg["label"] != "unlabeled"]
    eeg = eeg.loc[eeg["label"] != "everyone paired"]

    eeg.drop(
        [
            "indra_time",
            "Unnamed: 0",
            "browser_latency",
            "reading_time",
            "attention_esense",
            "meditation_esense",
            "updatedAt",
            "createdAt",
        ],
        axis=1,
        inplace=True,
    )

    eeg.reset_index(drop=True, inplace=True)
    eeg["raw_values"] = eeg["raw_values"].apply(convert_string_data_to_values)

    eeg = eeg.loc[eeg["signal_quality"] < QUALITY_THRESHOLD]
    eeg.replace(
                {
                    "label": {
                        "blink1": "blink",
                        "blink2": "blink",
                        "blink3": "blink",
                        "blink4": "blink",
                        "blink5": "blink",
                        "math1": "math",
                        "math2": "math",
                        "math3": "math",
                        "math4": "math",
                        "math5": "math",
                        "math6": "math",
                        "math7": "math",
                        "math8": "math",
                        "math9": "math",
                        "math10": "math",
                        "math11": "math",
                        "math12": "math",
                        "thinkOfItems-ver1": "thinkOfItems",
                        "thinkOfItems-ver2": "thinkOfItems",
                        "video-ver1": "video",
                        "video-ver2": "video",
                        "thinkOfItemsInstruction-ver1": "thinkOfItemsInstruction",
                        "thinkOfItemsInstruction-ver2": "thinkOfItemsInstruction",
                        "colorRound1-1": "colorRound1",
                        "colorRound1-2": "colorRound1",
                        "colorRound1-3": "colorRound1",
                        "colorRound1-4": "colorRound1",
                        "colorRound1-5": "colorRound1",
                        "colorRound1-6": "colorRound1",
                        "colorRound2-1": "colorRound2",
                        "colorRound2-2": "colorRound2",
                        "colorRound2-3": "colorRound2",
                        "colorRound2-4": "colorRound2",
                        "colorRound2-5": "colorRound2",
                        "colorRound2-6": "colorRound2",
                        "colorRound3-1": "colorRound3",
                        "colorRound3-2": "colorRound3",
                        "colorRound3-3": "colorRound3",
                        "colorRound3-4": "colorRound3",
                        "colorRound3-5": "colorRound3",
                        "colorRound3-6": "colorRound3",
                        "colorRound4-1": "colorRound4",
                        "colorRound4-2": "colorRound4",
                        "colorRound4-3": "colorRound4",
                        "colorRound4-4": "colorRound4",
                        "colorRound4-5": "colorRound4",
                        "colorRound4-6": "colorRound4",
                        "colorRound5-1": "colorRound5",
                        "colorRound5-2": "colorRound5",
                        "colorRound5-3": "colorRound5",
                        "colorRound5-4": "colorRound5",
                        "colorRound5-5": "colorRound5",
                        "colorRound5-6": "colorRound5",
                        "colorInstruction1": "colorInstruction",
                        "colorInstruction2": "colorInstruction",
                        "readyRound1": "readyRound",
                        "readyRound2": "readyRound",
                        "readyRound3": "readyRound",
                        "readyRound4": "readyRound",
                        "readyRound5": "readyRound",
                        "colorRound1": "colorRound",
                        "colorRound2": "colorRound",
                        "colorRound3": "colorRound",
                        "colorRound4": "colorRound",
                        "colorRound5": "colorRound",
                    }
                },
                inplace=True,
            )
    
    le = preprocessing.LabelEncoder()  # Generates a look-up table
    le.fit(eeg["label"])
    eeg["label"] = le.transform(eeg["label"])

    scaler = preprocessing.MinMaxScaler()
    series_list = [
        scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in eeg["raw_values"]
    ]
    labels_list = [i for i in eeg["label"]]

    X_train, X_test, y_train, y_test = train_test_split(
    series_list, labels_list, test_size=0.20, random_state=42, shuffle=True, stratify=labels_list
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(X_train.shape[0], 1, 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, 1, X_test.shape[1])
    np.savez(f"{filename.split('.')[0]}-train.npz", samples=X_train, classes=np.array(y_train))
    np.savez(f"{filename.split('.')[0]}-test.npz", samples=X_test, classes=np.array(y_test))

class EEGSynchtrain(Dataset):
    def __init__(self, filename="/home/bejar/ssdstorage/eeg-data-train.npz"):
        eeg = np.load(filename,allow_pickle=True)
        self.X_train = eeg['samples']
        self.y_train = eeg['classes']

        print(self.X_train.shape)
        print(np.unique(self.y_train))

    def __len__(self):
        return self.y_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class EEGSynchtest(Dataset):
    def __init__(self, filename="/home/bejar/ssdstorage/eeg-data-test.npz", val=True, random=0):
        eeg = np.load(filename,allow_pickle=True)
        X = eeg['samples']
        y = eeg['classes']

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random, shuffle=True, stratify=y
        )

        if val:
            self.X_train = X_train
            self.y_train = y_train 
        else:
            self.X_train = X_test
            self.y_train = y_test

        print(self.X_train.shape)
        print(np.unique(self.y_train))

    def __len__(self):
        return self.y_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
    

class EGGSynchgenerated(Dataset):
    def __init__(self, PATH='', DIR='', SAMP=0):
        ldata = np.load(f'{PATH}/{DIR}/Samples/sampled_data_{SAMP:05d}.npz')
        self.X_train = ldata['samples']
        self.y_train = ldata['classes']

        print(f'X_test shape is {self.X_train.shape}')
        print(f'y_test shape is {self.y_train.shape}')


    def __len__(self):
        return self.y_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx] 
    

class EEGSynchmixed(Dataset):
    def __init__(self, filename="/home/bejar/ssdstorage/eeg-data-train.npz", PATH='', DIR='', SAMP=0):
        eeg = np.load(filename,allow_pickle=True)
        self.X_train = eeg['samples']
        self.y_train = eeg['classes']

        ldata = np.load(f'{PATH}/{DIR}/Samples/sampled_data_{SAMP:05d}.npz')
        self.X_train = np.concatenate((self.X_train,ldata['samples']))
        self.y_train = np.concatenate((self.y_train,ldata['classes']))

        print(self.X_train.shape)
        print(np.unique(self.y_train))

    def __len__(self):
        return self.y_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
        
if __name__ == '__main__':
    preprocess()
    data = EEGSynchtrain(filename="/home/bejar/ssdstorage/eeg-data-train.npz")
    data = EEGSynchtrain(filename="/home/bejar/ssdstorage/eeg-data-test.npz")

