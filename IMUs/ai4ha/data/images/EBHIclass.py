import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class ClassBase(Dataset):
    def __init__(
        self,
        data_csv,
        data_root,
        size=None,
        random_crop=False,
        interpolation="bicubic",
        augmentation=False,
    ):
        self.data_csv = data_csv
        self.data_root = data_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)

        self.label_names = list(set([l.split("/")[-3] for l in self.image_paths]))

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
            "label": [l.split("/")[-3] for l in self.image_paths],
        }

        if augmentation:
            self.augmentation = albumentations.OneOf(
                [
                    albumentations.VerticalFlip(p=0.25),
                    albumentations.HorizontalFlip(p=0.25),
                    albumentations.RandomRotate90(p=0.25),
                ]
            )
        else:
            self.augmentation = None
        size = None if size is not None and size <= 0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=self.interpolation
            )
            self.segmentation_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_NEAREST
            )
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(
                    height=self.size, width=self.size
                )
            else:
                self.cropper = albumentations.RandomCrop(
                    height=self.size, width=self.size
                )
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
            processed = self.augmentation(image=processed["image"])

        example["image"] = (processed["image"] / 127.5 - 1.0).astype(np.float32)

        # Return a one hot encoding of the class
        example["class_label"] = self.label_names.index(example["label"])

        return example


class ExamplesTrain(ClassBase):
    def __init__(
        self,
        size=None,
        data_csv="/data/EBHI-SEG/EBHI-SEG-train.txt",
        data_root="/data",
        random_crop=False,
        interpolation="bicubic",
        augmentation=False,
    ):
        super().__init__(
            data_csv=data_csv,
            data_root=data_root,
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
            augmentation=augmentation,
        )


class ExamplesTest(ClassBase):
    def __init__(
        self,
        size=None,
        data_csv="/data/EBHI-SEG/EBHI-SEG-test.txt",
        data_root="/data",
        random_crop=False,
        interpolation="bicubic",
        augmentation=False,
    ):
        super().__init__(
            data_csv=data_csv,
            data_root=data_root,
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
        )


# class ExamplesDebug(ClassBase):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic", n_labels=2):
#         super().__init__(data_csv="/home/bejar/ssdstorage/lung_colon_image_set/colon_image_sets/ColonHisto.txt",
#                          data_root="/home/bejar/ssdstorage/lung_colon_image_set/colon_image_sets",
#                          size=size, random_crop=random_crop, interpolation=interpolation)

# if __name__ == '__main__':
#     ex = ExamplesDebug(size=256, interpolation='area')
#     print(ex[1]['label'])
#     for i in range(10):
#         print(i, ex[i]['label'])
