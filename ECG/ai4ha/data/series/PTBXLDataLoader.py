import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import numpy as np


class PTBXLtrain(Dataset):
    """ Loads the train data of PTBDB dataset """

    def __init__(self,
                 dir,
                 dataset='train',
                 padding=0,
                 padalgo='zero',
                 norm=False,
                 orig=True,
                 channel=None):
        self.dir = dir
        self.padalgo = padalgo
        self.dataset = dataset
        self.orig = orig
        print(f"Loading data from {self.dir}")
        if orig:
            datafiles = sorted(glob(f"{self.dir}/*_{dataset}.csv"))
            ldata = []
            for df in datafiles:
                data = pd.read_csv(df, header=None)
                tmp = data.iloc[:, :-1].values
                ldata.append(tmp.reshape(tmp.shape[0], 1, tmp.shape[1]))
            self.X_train = np.concatenate(ldata, axis=1)
            self.y_train = np.array(data.iloc[:,
                                              -1].astype('category').cat.codes)
        else:
            data = np.load(f"{self.dir}/PTBXL_{dataset}.npz")
            self.X_train = data['X_train']
            self.y_train = np.array(data['y_train'])

        # Put zeros on nan values
        np.nan_to_num(self.X_train, copy=False)

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

        if channel is not None:
            self.X_train = self.X_train[:, channel, :]
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1,
                                                self.X_train.shape[1])
        if norm:
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))

        print(f'PTBXL {self.dataset} NPZ {self.orig}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))
        print(f'PAD={padding} PADALGO={padalgo}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def save_data(self):
        np.savez_compressed(f"{self.dir}/PTBXL_{self.dataset}.npz",
                            X_train=self.X_train,
                            y_train=self.y_train)


if __name__ == "__main__":
    train = PTBXLtrain("/home/bejar/bsc/Data/ptbxl")
    # train_loader = DataLoader(train, batch_size=64, shuffle=True)
    # for i, (X, y) in enumerate(train_loader):
    #     print(X.shape, y.shape)
    #     if i == 10:
    #         break
