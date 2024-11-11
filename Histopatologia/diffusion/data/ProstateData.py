import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch

Image.MAX_IMAGE_PIXELS = None


def load_mask(mask_path, provider):
    mask = Image.open(mask_path) 
    if not mask.mode == "RGB":
            mask = mask.convert("RGB")
    mask = np.array(mask).astype(np.uint8)

    mask = mask[..., 0]
    if provider == 'radboud': 
        mask[mask <= 2] = 0 
        mask[mask > 2] = 1
    if provider == 'karolinska':
        mask[mask <= 1] = 0
        mask[mask == 2] = 1

    return np.expand_dims(mask, axis=-1)


class ProstateDataset(Dataset):
    def __init__(self,
                 data_csv, data_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=1,
                 augmentation=False,
                 ):
        self.n_labels = n_labels
        self.data_csv = data_csv
        self.data_root = data_root

        self.image_paths = []
        self.isup_grade = []
        self.provider = []
        with open(self.data_csv, "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_paths.append(line.split(',')[0])
                self.provider.append(line.split(',')[1])
                self.isup_grade.append(line.split(',')[2])
             
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l+'.png' for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l+'.png')
                           for l in self.image_paths],
            "class_label": self.isup_grade, 
            "provider": self.provider
        }

        size = None if size is not None and size<=0 else size
        self.size = size

        if augmentation:
            self.augmentation = albumentations.OneOf(
                    [albumentations.VerticalFlip(p=0.25),
                    albumentations.HorizontalFlip(p=0.25),
                     albumentations.RandomRotate90(p=0.25)
                    ])
        else:
            self.augmentation = None

        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        if self.size is not None:
            processed = self.preprocessor(image=image)
        else:
            processed = {"image": image}

        if self.augmentation is not None:
            processed = self.augmentation(image=processed['image'])       

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32) #/127.5 - 1.0
        example["image"] = np.moveaxis(example["image"], -1, 0)

        return example


class ExamplesTrain(ProstateDataset):
    def __init__(self, 
                 size=None, 
                 data_csv="/home/aromero/Desktop/Datasets/PANDA/train.csv",
                 data_root="/home/aromero/Desktop/Datasets/PANDA/images",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                augmentation=False):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         augmentation=augmentation)


class ExamplesTest(ProstateDataset):
    def __init__(self, 
                 size=None, 
                 data_csv="/home/aromero/Desktop/Datasets/PANDA/train.csv",
                 data_root="/home/aromero/Desktop/Datasets/PANDA/images",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                 augmentation=False):
        super().__init__(size=size, 
                         data_csv=data_csv,
                         data_root=data_root,
                         n_labels=n_labels,
                         random_crop=random_crop, 
                         interpolation=interpolation,
                         augmentation=augmentation)


class ExamplesDebug(ProstateDataset):
    def __init__(self, size=None, random_crop=False, 
                 interpolation="bicubic", n_labels=2):
        super().__init__(data_csv="/home/aromero/Desktop/Datasets/PANDA/train.csv",
                         data_root="/home/aromero/Desktop/Datasets/PANDA/images",
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation)


class ExamplesVAE(Dataset):
    def __init__(self,
                model,
                size,
                root='/gpfs/projects/bsc70/bsc70174/Data/PANDAGen',
                normalization=1.):
        base_path = f'{root}/{model}/{size}/'
        self.images = [os.path.join(base_path,  d)
                       for d in sorted(os.listdir(base_path))]
        self.normalization = normalization

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]

        example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization))}
        return example



"""
if __name__ == '__main__':
    ex = ExamplesDebug(size=512, n_labels=2, interpolation='bicubic')
    img = np.moveaxis(ex[11]['image'], 0, -1)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    print(img.shape)

    seg = np.moveaxis(ex[11]['segmentation'], 0, -1)
    plt.imshow(seg, cmap='gray')
    plt.axis('off')
    plt.show()

    print(ex[1]['segmentation'].shape)
    print(ex[0]['segmentation'].shape)  
    for i in range(10):
        print(i, ex[i]['segmentation'].shape)
        print(i, ex[i]['imageseg'].sape)
""" 