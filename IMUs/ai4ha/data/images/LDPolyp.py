import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
from numpy.random import rand
from glob import glob


def load_bbox(fn, shape):
    bbfile = open(fn)
    nbbs = int(bbfile.readline())
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)  # initialize mask
    for bb in range(nbbs):
        x1, y1, x2, y2 = bbfile.readline().split(" ")[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp
        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp
        mask[y1:y2, x1:x2] = 1  # fill with white pixels
    bbfile.close()
    return mask


class SegmentationBase(Dataset):

    def __init__(
        self,
        data_csv,
        data_root,
        dataset,
        size=None,
        random_crop=False,
        interpolation="bicubic",
        n_labels=1,
        shift_segmentation=False,
        augmentation=False,
    ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation

        if type(data_csv) is not list:
            data_csv = [data_csv]

        if type(dataset) is not list:
            dataset = [dataset]

        self.labels = {
            "relative_file_path_": [],
            "file_path_": [],
            "segmentation_path_": []
        }

        self._length = 0
        for csv, dset in zip(data_csv, dataset):
            self.data_csv = f"{data_root}/{csv}"
            self.data_root = f'{data_root}/{dset}/images/'
            self.segmentation_root = f'{data_root}/{dset}/bbox/'
            with open(f"{self.data_csv}", "r") as f:
                self.image_paths = f.read().splitlines()
            self.labels["relative_file_path_"] += [l for l in self.image_paths]
            self.labels["file_path_"] += [
                os.path.join(self.data_root, l) for l in self.image_paths
            ]
            self.labels["segmentation_path_"] += [
                os.path.join(self.segmentation_root, l.replace(".jpg", ".txt"))
                for l in self.image_paths
            ]
            # self.labels = {
            #     "relative_file_path_": [l for l in self.image_paths],
            #     "file_path_":
            #     [os.path.join(self.data_root, l) for l in self.image_paths],
            #     "segmentation_path_": [
            #         os.path.join(self.segmentation_root,
            #                      l.replace(".jpg", ".txt"))
            #         for l in self.image_paths
            #     ],
            # }

            self._length += len(self.image_paths)

        size = None if size is not None and size <= 0 else size
        self.size = size

        if augmentation:
            self.augmentation = albumentations.OneOf([
                albumentations.VerticalFlip(p=0.25),
                albumentations.HorizontalFlip(p=0.25),
                albumentations.RandomRotate90(p=0.25),
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
                "lanczos": cv2.INTER_LANCZOS4,
            }[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(
                max_size=self.size, interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,
                                                         width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,
                                                         width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        segmentation = load_bbox(example["segmentation_path_"], image.shape)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation + 1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(
                image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)
        else:
            processed = {"image": image, "mask": segmentation}

        if self.augmentation is not None:
            processed = self.augmentation(image=processed["image"],
                                          mask=processed["mask"])

        example["image"] = (processed["image"] / 127.5 - 1.0).astype(
            np.float32)
        segmentation = processed["mask"]

        onehot = np.abs(
            np.eye(self.n_labels)[segmentation].astype(np.float32) - 1)
        example["segmentation"] = onehot
        s_onehot = ((onehot / np.max(onehot)) * 2) - 1
        example["imageseg"] = np.concatenate((example["image"], s_onehot),
                                             axis=2)
        example["aesegmentation"] = s_onehot

        return example


class ExamplesTrain(SegmentationBase):
    def __init__(
        self,
        size=None,
        data_csv="LDPolip_train.txt",
        data_root="/LDPolyp/",
        dataset="train",
        random_crop=False,
        interpolation="bicubic",
        n_labels=2,
        augmentation=False,
    ):
        super().__init__(
            data_csv=data_csv,
            data_root=data_root,
            dataset=dataset,
            n_labels=n_labels,
            size=size,
            random_crop=random_crop,
            interpolation=interpolation,
            augmentation=augmentation,
        )


class ExamplesTest(SegmentationBase):
    def __init__(
        self,
        size=None,
        data_csv="/data/LDPolyp/LDPolip_test.txt",
        data_root="/LDPolyp/",
        dataset="test",
        random_crop=False,
        interpolation="bicubic",
        n_labels=2,
        augmentation=False,
    ):
        super().__init__(
            size=size,
            data_csv=data_csv,
            data_root=data_root,
            dataset=dataset,
            n_labels=n_labels,
            random_crop=random_crop,
            interpolation=interpolation,
            augmentation=augmentation,
        )


class Examples(SegmentationBase):
    def __init__(
        self,
        size=None,
        data_csv="/data/LDPolyp/LDPolip_test.txt",
        data_root="/LDPolyp/",
        dataset="test",
        random_crop=False,
        interpolation="bicubic",
        n_labels=2,
        augmentation=False,
    ):
        super().__init__(
            size=size,
            data_csv=data_csv,
            data_root=data_root,
            dataset=dataset,
            n_labels=n_labels,
            random_crop=random_crop,
            interpolation=interpolation,
            augmentation=augmentation,
        )


class ExamplesVAE(Dataset):
    def __init__(
        self,
        model="AEKL",
        size=256,
        encoder="AllAEKL64",
        data_root="/gpfs/projects/bsc70/bsc70642/Data/LDPolyp",
        normalization=1.0,
        segmentation=False,
        augmentation=False,
        augprob=0.5,
    ):
        base_path = f"{data_root}/{model}/{size}/{encoder}/"
        if augmentation:
            base_path += "aug/"
            self.images = [
                os.path.join(os.path.join(base_path, "images"), d)
                for d in sorted(os.listdir(os.path.join(base_path, "images")))
                if "orig" in d
            ]
            if segmentation:
                self.segmentations = [
                    os.path.join(os.path.join(base_path, "masks"), d)
                    for d in sorted(os.listdir(os.path.join(base_path, "masks")))
                    if "orig" in d
                ]
        else:
            self.images = [
                os.path.join(os.path.join(base_path, "images"), d)
                for d in sorted(os.listdir(os.path.join(base_path, "images")))
            ]

            if segmentation:
                self.segmentations = [
                    os.path.join(os.path.join(base_path, "masks"), d)
                    for d in sorted(os.listdir(os.path.join(base_path, "masks")))
                ]

        self.normalization = normalization
        self.seg = segmentation
        self.augprob = augprob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ntrans = 5
        augdiv = self.augprob / ntrans
        ltrans = ["vflip", "hflip", "rot90", "rot180", "rot270", "orig"]
        vtrans = []
        for i in range(ntrans):
            vtrans.append([augdiv * (i + 1), ltrans[i]])
        vtrans.append([1, "orig"])

        rep = "orig"
        for i in range(ntrans + 1):
            if rand(1)[0] < vtrans[i][0]:
                rep = vtrans[i][1]
                break

        img_path = self.images[idx].replace("orig", rep)
        # print(img_path)
        if self.seg:
            image = torch.tensor(np.load(img_path) / (1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace("orig", rep)
            onehot = torch.tensor(np.load(mask_path))
            # s_onehot = ((onehot/onehot.max())*2)-1
            example = {
                "image": image,
                "segmentation": onehot,
                # "imageseg": np.concatenate((image, s_onehot), axis=0)
            }
        else:
            example = {
                "image": torch.tensor(np.load(img_path) / (1.0 * self.normalization))
            }
        return example


class LDPolypGenDataset(Dataset):

    def __init__(self, data_root='/data/', size=480):
        self.path = data_root
        self.size = size

        self.images = []
        self.masks = []
        self.images = sorted(glob(f"{self.path}/images/*.png"))
        self.masks = sorted(glob(f"{self.path}/masks/*.png"))
        self.bboxes = sorted(glob(f"{self.path}/labels/*.txt"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        segmentation_path = self.masks[idx]
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        mask = Image.open(segmentation_path)
        if not mask.mode == "L":
            mask = mask.convert("L")
        mask = np.array(mask).astype(np.uint8)

        bbox_path = os.path.join(
            self.path,
            "labels",
            self.bboxes[idx],
        )
        bbfile = open(bbox_path, "r")
        bbox = [b for b in bbfile]
        bbox = [b.split(" ") for b in bbox]
        bbox = [
            torch.tensor([
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
                float(b[4])
            ]) for b in bbox
        ]
        bbfile.close()

        image = (image / 127.5 - 1.0).astype(np.float32)
        example = {
            "image": torch.tensor(image),
            "segmentation": torch.tensor(mask).unsqueeze(2),
            # "bbox": bbox,
        }

        return example


if __name__ == "__main__":
    from skimage.measure import block_reduce

    ex = ExamplesDebug(size=256, n_labels=2, interpolation="area")
    print(ex[1]["segmentation"].shape)
    # print(ex[0]['segmentation'].shape)
    for i in range(10):
        print(i, ex[i]["segmentation"].shape)
        print(i, ex[i]["imageseg"].shape)
        print(ex[i]["segmentation"])
