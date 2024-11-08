import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from glob import glob


class ColonDataset(Dataset):

    def __init__(self, data_root='/data/', size=480):
        self.path = data_root
        self.size = size

        self.images = []
        self.masks = []
        self.images = sorted(glob(f"{self.path}/images/*.jpg"))
        self.masks = sorted(glob(f"{self.path}/masks/*.npy"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        segmentation_path = self.masks[idx]
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        image = (image / 127.5 - 1.0).astype(np.float32)
        example = {
            "image": torch.tensor(image),
            "segmentation":
            torch.tensor(np.load(segmentation_path)).unsqueeze(2)
        }
        return example
