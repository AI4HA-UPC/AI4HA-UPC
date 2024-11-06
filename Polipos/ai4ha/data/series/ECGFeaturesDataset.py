from torch.utils.data import Dataset
import numpy as np
import torch

class ECGFeaturesDataset(Dataset):
    """ Loads ECG Featuresdatasets """

    def __init__(self,
                 dir,
                 dataset='train',
                 norm=False,):
        self.dir = dir
        self.dataset = dataset
        print(f"Loading data from {self.dir}")
        data = np.load(f"{self.dir}/{dataset}.npy")
        self.X_train = data[:, :-1]
        self.y_train = data[:, -1]

        if norm:
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))

        print(f'{self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return torch.tensor(self.X_train[idx]).unsqueeze(0), self.y_train[idx]