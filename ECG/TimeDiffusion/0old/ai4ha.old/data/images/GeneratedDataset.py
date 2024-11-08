import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy

DATASETS = ["SUNColon", "LDPolyp", "SUNLDPolyp", "ColonPolyp", "MultiColon"]


class GeneratedDataset(Dataset):

    def __init__(self,
                 root,
                 dataset="LDPolyp",
                 size=640,
                 aug=50,
                 steps=40,
                 bbox=True,
                 mask=False):
        self.root = root
        self.dataset = dataset
        self.aug = aug
        self.steps = steps
        self.bbox = bbox
        self.mask = mask
        self.images = []
        self.masks = []
        self.bboxes = []
        self.images = sorted(
            os.listdir(
                os.path.join(self.root, self.dataset, f"{aug:02d}", f"{steps}",
                             "images")))
        if mask:
            self.masks = sorted(
                os.listdir(
                    os.path.join(self.root, self.dataset, f"{aug:02d}",
                                 f"{steps}", "masks")))
        if bbox:
            self.bboxes = sorted(
                os.listdir(
                    os.path.join(self.root, self.dataset, f"{aug:02d}",
                                 f"{steps}", "labels")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root,
            self.dataset,
            f"{self.aug:02d}",
            f"{self.steps}",
            "images",
            self.images[idx],
        )
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = numpy.array(image).astype(numpy.uint8)

        if self.mask:
            mask_path = os.path.join(
                self.root,
                self.dataset,
                f"{self.aug:02d}",
                f"{self.steps}",
                "masks",
                self.masks[idx],
            )
            mask = Image.open(mask_path)
            if not mask.mode == "L":
                mask = mask.convert("L")
            mask = numpy.array(mask).astype(numpy.uint8)

        if self.bbox:
            bbox_path = os.path.join(
                self.root,
                self.dataset,
                f"{self.aug:02d}",
                f"{self.steps}",
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
        if self.mask and self.bbox:
            example = {
                "image": torch.tensor(image),
                "mask": torch.tensor(mask),
                "bbox": bbox,
            }
        elif self.mask:
            example = {
                "image": torch.tensor(image),
                "mask": torch.tensor(mask)
            }
        elif self.bbox:
            example = {"image": torch.tensor(image), "bbox": bbox}
        else:
            example = {"image": torch.tensor(image)}

        return example
