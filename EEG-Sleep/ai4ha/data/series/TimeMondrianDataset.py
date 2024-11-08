from torch.utils.data import Dataset
import numpy as np


class TimeMondrianDataset(Dataset):
    """ Loads the Time Mondrian dataset 
        random Mondrian process data for testing time series models
    """

    def __init__(self,
                 dir,
                 classes=7,
                 sampling=256,
                 cp=0.1,
                 cd=0.1,
                 sigma=2,
                 norm=1,
                 padding=0,
                 padalgo='zero'):
        """
        Loads the Time Mondrian dataset
        """
        self.dir = dir

        self.X_train = np.load(
            f"{self.dir}/TimeMondrian_{sampling}_{cp}_{cd}_{sigma}.npy")
        self.y_train = np.array([classes] * self.X_train.shape[0], dtype=int)

        if padding > 0:
            if padalgo == 'zero':
                tmp = np.zeros((self.X_train.shape[0], self.X_train.shape[1],
                                self.X_train.shape[2] + padding))

                tmp[:, :, :self.X_train.shape[2]] = self.X_train
                self.X_train = tmp
            elif padalgo == 'repeat':
                tmp = np.zeros((self.X_train.shape[0], self.X_train.shape[1],
                                self.X_train.shape[2] + padding))
                tmp[:, :, :self.X_train.shape[2]] = self.X_train
                for i in range(self.X_train.shape[2],
                               self.X_train.shape[2] + padding):
                    tmp[:, :, i] = self.X_train[:, :,
                                                self.X_train.shape[2] - 1]
                self.X_train = tmp

        if norm is not None:  # -norm:norm normalization
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))
            self.X_train = (2*norm) * self.X_train - norm
        print(f'TimeMondrian')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
