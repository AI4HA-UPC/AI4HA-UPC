from torch.utils.data import Dataset
import numpy as np


class MareaDataset(Dataset):
    """ Loads the data Marea dataset """

    def __init__(self,
                 dir,
                 classes=7,
                 sampling=256,
                 window=256,
                 dataset='train',
                 padding=0,
                 padalgo='zero',
                 norm=False):
        self.dir = dir
        self.padalgo = padalgo
        self.dataset = dataset

        if classes == 7:
            tmp1 = np.load(
                f"{self.dir}/marea_indoor_{sampling}_{window}_{dataset}.npz")
            tmp2 = np.load(
                f"{self.dir}/marea_outdoor_{sampling}_{window}_{dataset}.npz")
            self.X_train = np.concatenate([tmp1['data'], tmp2['data']])
            self.y_train = np.concatenate([tmp1['labels'], tmp2['labels']])
        elif classes == 4:
            tmp1 = np.load(
                f"{self.dir}/marea4_indoor_{sampling}_{window}_{dataset}.npz")
            tmp2 = np.load(
                f"{self.dir}/marea4_outdoor_{sampling}_{window}_{dataset}.npz")
            self.X_train = np.concatenate([tmp1['data'], tmp2['data']])
            self.y_train = np.concatenate([tmp1['labels'], tmp2['labels']])
        else:
            raise ValueError("Classes must be 4 or 7")

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

        if norm:  # -1:1 normalization
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))
            self.X_train = 2 * self.X_train - 1
        print(f'MAREA {self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))
        print(f'PAD={padding} PADALGO={padalgo}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class MareaGenDataset(Dataset):
    """ Loads the data Marea dataset """

    def __init__(self,
                 dir,
                 classes=7,
                 sampling=256,
                 window=256,
                 dataset='train',
                 gdataset='diff',
                 gsize=256,
                 mondrian=None,
                 padding=0,
                 padalgo='zero',
                 norm=False,
                 only_gen=False):
        self.dir = dir
        self.padalgo = padalgo
        self.dataset = dataset

        if classes == 7:
            tmp1 = np.load(
                f"{self.dir}/marea_indoor_{sampling}_{window}_{dataset}.npz")
            tmp2 = np.load(
                f"{self.dir}/marea_outdoor_{sampling}__{window}_{dataset}.npz")
            self.X_train = np.concatenate([tmp1['data'], tmp2['data']])
            self.y_train = np.concatenate([tmp1['labels'], tmp2['labels']])
        elif classes == 4:
            tmp1 = np.load(
                f"{self.dir}/marea4_indoor_{sampling}_{window}_{dataset}.npz")
            tmp2 = np.load(
                f"{self.dir}/marea4_outdoor_{sampling}_{window}_{dataset}.npz")
            self.X_train = np.concatenate([tmp1['data'], tmp2['data']])
            self.y_train = np.concatenate([tmp1['labels'], tmp2['labels']])
        else:
            raise ValueError("Classes must be 4 or 7")

        if mondrian is None:
            gen_samples = np.load(
                f'{self.dir}/marea{classes}_{sampling}_{window}_l{gsize}_{gdataset}.npz'
            )
            X_gen = gen_samples['samples']
            y_gen = gen_samples['classes']
        else:
            gen_samples = np.load(
                f'{self.dir}/marea{classes}_{sampling}_{window}_l{gsize}_{gdataset}_m{mondrian}.npz'
            )
            X_gen = gen_samples['samples'][gen_samples['classes'] != classes]
            y_gen = gen_samples['classes'][gen_samples['classes'] != classes]

        if norm:  # -1:1 normalization
            self.X_train = (self.X_train - np.min(self.X_train)) / \
                (np.max(self.X_train) - np.min(self.X_train))
            self.X_train = 2 * self.X_train - 1
        else:
            # Generated samples are normalized
            X_gen = (X_gen + 1) / 2
            X_gen = (
                X_gen *
                (np.max(self.X_train) - np.min(self.X_train))) + np.min(
                    self.X_train)

        if only_gen:
            self.X_train = X_gen
            self.y_train = y_gen
        else:
            self.X_train = np.concatenate(
                [self.X_train, X_gen])
            self.y_train = np.concatenate(
                [self.y_train, y_gen])

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

        print(f'MAREA {self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))
        print(f'PAD={padding} PADALGO={padalgo}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
