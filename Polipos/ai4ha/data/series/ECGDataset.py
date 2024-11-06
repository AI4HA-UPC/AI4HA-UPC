from glob import glob
from torch.utils.data import Dataset
import numpy as np


class ECGDataset(Dataset):
    """ Loads ECG datasets """

    def __init__(self,
                 dir,
                 dataset='train',
                 padding=0,
                 padalgo='zero',
                 norm=False,
                 channel=None,
                 sampling_smoothing=None,
                 sampling_smoothing_value=5000):
        self.dir = dir
        self.padalgo = padalgo
        self.dataset = dataset
        print(f"Loading data from {self.dir}")
        datafiles = sorted(glob(f"{self.dir}/*_{dataset}.npz"))
        data = np.load(datafiles[0])
        try:
            self.X_train = data['data']
            self.y_train = data['labels']
        except:
            self.X_train = data['samples']
            self.y_train = data['classes']
            
        self.weights = None
        self.max_val = np.max(self.X_train)
        self.min_val = np.min(self.X_train)

        # Computes the weights for the samples as as the inverse/direct frequency of the classes appling laplace smoothing
        if sampling_smoothing is not None:
            counts = np.unique(self.y_train, return_counts=True)[1]
            if sampling_smoothing == 'inverse':
                class_weights = (np.sum(counts) +
                                 (len(counts) * sampling_smoothing_value)) / (
                                     counts + sampling_smoothing_value)
                self.weights = [class_weights[i] for i in self.y_train]
            else:
                class_weights = (counts + sampling_smoothing_value) / (
                    np.sum(counts) + (len(counts) * sampling_smoothing_value))
                self.weights = [class_weights[i] for i in self.y_train]
        # Put zeros on nan values
        np.nan_to_num(self.X_train, copy=False)
        if padding > 0:
            if padalgo == 'zero':
                self.X_train = np.concatenate([
                    self.X_train,
                    np.zeros((self.X_train.shape[0], self.X_train.shape[1],
                              padding))
                ],
                                              axis=2)
            elif padalgo == 'zero-s':
                self.X_train = np.concatenate([
                    np.zeros((self.X_train.shape[0], self.X_train.shape[1],
                              padding // 2)), self.X_train,
                    np.zeros((self.X_train.shape[0], self.X_train.shape[1],
                              padding // 2))
                ],
                                              axis=2)
            elif padalgo == 'repeat':
                tmp = np.repeat(self.X_train[:, :, -1],
                                padding).reshape(self.X_train.shape[0],
                                                 self.X_train.shape[1],
                                                 padding)
                self.X_train = np.concatenate([self.X_train, tmp], axis=2)
            elif padalgo == 'repeat-s':
                tmp = np.repeat(self.X_train[:, :, -1],
                                padding // 2).reshape(self.X_train.shape[0],
                                                      self.X_train.shape[1],
                                                      padding // 2)
                tmp2 = np.repeat(self.X_train[:, :, 0],
                                 padding // 2).reshape(self.X_train.shape[0],
                                                       self.X_train.shape[1],
                                                       padding // 2)
                self.X_train = np.concatenate([tmp2, self.X_train, tmp],
                                              axis=2)
            elif padalgo == 'mirror':
                tmp = np.flip(self.X_train[:, :, -padding:], axis=2)
                self.X_train = np.concatenate([self.X_train, tmp], axis=2)
            elif padalgo == 'mirror-s':
                tmp = np.flip(self.X_train[:, :, -padding // 2:], axis=2)
                tmp2 = np.flip(self.X_train[:, :, :padding // 2], axis=2)
                self.X_train = np.concatenate([tmp2, self.X_train, tmp],
                                              axis=2)
            elif padalgo == 'random':
                tmp = np.random.randn(self.X_train.shape[0],
                                      self.X_train.shape[1], padding)
                self.X_train = np.concatenate([self.X_train, tmp], axis=2)
            elif padalgo == 'random-s':
                tmp = np.random.randn(self.X_train.shape[0],
                                      self.X_train.shape[1], padding // 2)
                self.X_train = np.concatenate([tmp, self.X_train, tmp], axis=2)
            elif padalgo == 'circular-s':
                tmp = np.concatenate([
                    self.X_train[:, :, -padding // 2:], self.X_train,
                    self.X_train[:, :, :padding // 2]
                ],
                                     axis=2)
                self.X_train = tmp
            elif padalgo == 'mirror-double':
                tmp = np.flip(self.X_train[:, :, :], axis=2)
                self.X_train = np.concatenate([self.X_train, tmp], axis=2)
        # if padding > 0:
        #     if padalgo == 'zero':
        #         tmp = np.zeros((self.X_train.shape[0], self.X_train.shape[1],
        #                         self.X_train.shape[2] + padding))

        #         tmp[:, :, :self.X_train.shape[2]] = self.X_train
        #         self.X_train = tmp
        #     elif padalgo == 'repeat':
        #         tmp = np.zeros((self.X_train.shape[0], self.X_train.shape[1],
        #                         self.X_train.shape[2] + padding))
        #         tmp[:, :, :self.X_train.shape[2]] = self.X_train
        #         for i in range(self.X_train.shape[2],
        #                        self.X_train.shape[2] + padding):
        #             tmp[:, :, i] = self.X_train[:, :,
        #                                         self.X_train.shape[2] - 1]
        #         self.X_train = tmp
        #     elif padalgo == 'mirror':
        #         tmp = np.zeros((self.X_train.shape[0], self.X_train.shape[1],
        #                         self.X_train.shape[2] + padding))
        #         tmp[:, :, :self.X_train.shape[2]] = self.X_train
        #         for i in range(self.X_train.shape[2],
        #                        self.X_train.shape[2] + padding):
        #             tmp[:, :, i] = self.X_train[:, :,
        #                                         self.X_train.shape[2] - 1 - (
        #                                             i - self.X_train.shape[2])]
        #         self.X_train = tmp

        if channel is not None:
            self.X_train = self.X_train[:, channel, :]
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1,
                                                self.X_train.shape[1])
        if norm:
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))

        print(f'{self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))
        print(
            f'PAD={padding} PADALGO={padalgo} NORM={norm} CHANNEL={self.X_train.shape[1]}/{channel}'
        )

        if sampling_smoothing is not None:
            print(f'Sampling smoothing Weights for classes: {class_weights}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


if __name__ == "__main__":
    train = ECGDataset("/home/bejar/bsc/Data/PTBXL", dataset="train")
    # train_loader = DataLoader(train, batch_size=64, shuffle=True)
    # for i, (X, y) in enumerate(train_loader):
    #     print(X.shape, y.shape)
    #     if i == 10:
    #         break
