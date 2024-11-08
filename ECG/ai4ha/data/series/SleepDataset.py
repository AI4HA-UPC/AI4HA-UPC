import numpy as np
from glob import glob
from torch.utils.data import Dataset


class SleepDataset(Dataset):

    def __init__(
        self,
        dir,
        dataset='train',
        padding=0,
        padalgo='zero',
        norm=None,
    ):

        self.dir = dir
        self.padalgo = padalgo
        if dataset in ['train', 'val', 'test']:
            self.dataset = dataset
            print(f"Loading data from {self.dir}")
            subjects = sorted(glob(f'{self.dir}/{dataset}_set/*.npz'))
            dataset = []
            labels = []
            for s in subjects:
                data = np.load(s, allow_pickle=True)
                dataset.append(np.swapaxes(data['x'], 1, 2))
                labels.append(data['y'])
        elif dataset == 'all':
            self.dataset = 'all'
            print(f"Loading data from {self.dir}")
            subjects = sorted(glob(f'{self.dir}/*/*.npz'))
            dataset = []
            labels = []
            for s in subjects:
                data = np.load(s, allow_pickle=True)
                dataset.append(np.swapaxes(data['x'], 1, 2))
                labels.append(data['y'])
        else:
            raise ValueError(f"dataset must be 'train', 'val', 'test' or 'all'")
            
        self.X_train, self.y_train = np.concatenate(dataset), np.concatenate(
            labels)
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
        if norm == 'scale1':
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))
        elif norm == 'scale2':
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train)) * 2 - 1
        elif norm == 'zscore':
            self.X_train = (self.X_train - np.mean(self.X_train)) / \
                np.std(self.X_train)

        print(f'{self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))
        print(f'PAD={padding} PADALGO={padalgo} NORM={norm}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
