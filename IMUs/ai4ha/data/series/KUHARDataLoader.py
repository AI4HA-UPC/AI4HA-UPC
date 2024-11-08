import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class KUHARtrain(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """ """_Data Loader for el KU-HAR dataset_

    KU-HAR: Human Activity Recognition Dataset (v 1.0)
    https://www.kaggle.com/datasets/niloy333/kuhar


    Data corresponding to the KU-HAR_time_domain_subsamples_20750x300.csv file

     20750 subsamples extracted from the 1945 collected samples provided in a single .csv file.
     Each of them contains 3 seconds of non-overlapping data of the corresponding activity.
     Arrangement of information:

    Col. 1-300, 301-600, 601-900-> Acc.meter X, Y, Z axes readings
    Col. 901-1200, 1201-1500, 1501-1800-> Gyro X, Y, Z axes readings
    Col. 1801-> Class ID (0 to 17, in the order mentioned above)
    Col. 1802-> length of the each channel data in the subsample
    Col. 1803-> serial no. of the subsample

     examplesx1xchannelsxsignal length

    @2023-03-30 Normalizing channels to 0-1 range

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self,
                 filename='./UID.csv',
                 channels=6,
                 padding=0,
                 padalgo='zero'):
        data_train = pd.read_csv(filename, header=None)
        self.y_train = data_train.iloc[:, -3].values
        tmp = data_train.iloc[:, :-3].values
        length = (tmp.shape[1]) // channels
        if (tmp.shape[1] / channels) > length:
            raise NameError(
                "Wrong  number of channels or uneven signal length")

        tmp = tmp.reshape(tmp.shape[0], channels, length)

        # Robust scaling
        for i in range(channels):
            qu = np.quantile(tmp[:, i, :], 0.99)
            qd = np.quantile(tmp[:, i, :], 0.01)
            tmp[:, i, :] = np.clip(tmp[:, i, :], qd, qu)
            tmp[:, i, :] = (tmp[:, i, :] - np.min(tmp[:, i, :])) / (
                np.max(tmp[:, i, :]) - np.min(tmp[:, i, :]))

        if padding == 0:
            self.X_train = tmp.reshape(tmp.shape[0], channels, length)
        elif padalgo == 'zero':
            self.X_train = np.zeros((tmp.shape[0], channels, length + padding))
            self.X_train[:, :, :length] = tmp.reshape(tmp.shape[0], channels,
                                                      length)
        elif padalgo == 'repeat':
            self.X_train = np.zeros((tmp.shape[0], channels, length + padding))
            self.X_train[:, :, :length] = tmp.reshape(tmp.shape[0], channels,
                                                      length)
            for i in range(length, length + padding):
                self.X_train[:, :, i] = self.X_train[:, :, length - 1]

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(f'PAD={padding} PADALGO={padalgo}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


if __name__ == '__main__':
    data = KUHARtrain(filename="/home/bejar/ssdstorage/KU-HAR/KU-HAR.csv",
                      channels=6)

    print(data[0][0].shape)
