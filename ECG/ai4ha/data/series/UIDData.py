import os
import numpy as np
import pandas as pd
import sys

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class UIDtrain(Dataset):
    """_Data Loader for el UIWAD dataset_

    User Identification From Walking Activity Data Set
    https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity

    Original data joined in one file tagging series of 300 time steps with the user
    Three channels corresponding to X-Y-Z axis acceleration

    Data is adapted to the format to TTS-cGAN

     1x1xchannelsxsignal length

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, filename="./UID.csv", channels=3):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:, -1].values
        tmp = data_train.iloc[:, :-1].values
        length = (tmp.shape[1]) // channels
        if (tmp.shape[1] / channels) > length:
            raise NameError("Wrong  number of channels or uneven signal length")

        length = (tmp.shape[1]) // channels
        nex = tmp.shape[0]
        tmp = tmp.reshape(nex, channels, length)
        tmp = np.swapaxes(tmp, 1, 2)
        tmp = tmp.reshape(nex * length, channels)
        tmp = MinMaxScaler().fit_transform(tmp)
        tmp = tmp.reshape(nex, length, channels)
        tmp = np.swapaxes(tmp, 1, 2)

        self.X_train = tmp.reshape(nex, 1, channels, length)

        print(f"X_train shape is {self.X_train.shape}")
        print(f"y_train shape is {self.y_train.shape}")

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


if __name__ == "__main__":
    data = UIDtrain(filename="/home/bejar/ssdstorage/UID/UID.csv", channels=3)
    print(data[0][0].shape)
