import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
from numpy.random import rand
from tqdm import tqdm 


# ------------------------------------- #
# Dataset class for the original images 
# ------------------------------------- #
class ProstateDatasetImages(Dataset):
    def __init__(self,
                data_csv,
                data_root,
                size=None,
                interpolation="bicubic",
                augmentation=False):
    
        self.data_csv = data_csv
        self.data_root = data_root
        
        self.image_ids = []
        self.gleason_list = []
        with open(self.data_csv, "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_ids.append(line.split(',')[0])
                self.gleason_list.append(line.split(',')[1])
             
        self._length = len(self.image_ids)
        self.labels = {
            "image_name": self.image_ids,
            "relative_file_path_": [l + '.png' for l in self.image_ids],
            "file_path_": [os.path.join(self.data_root, l + '.png') for l in self.image_ids],
            "gleason": self.gleason_list
        }

        self.size = size if size is None or size > 0 else None

        if augmentation:
            self.augmentation = albumentations.OneOf([
                albumentations.VerticalFlip(p=0.25),
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
                "lanczos": cv2.INTER_LANCZOS4
            }[self.interpolation]
            self.image_rescaler = albumentations.Resize(height=self.size, width=self.size, interpolation=self.interpolation)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        try:
            image = Image.open(example["file_path_"])
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
        except (OSError, ValueError) as e:
            print(f"Skipping corrupted image {example['file_path_']}: {e}")
            # Return a dummy example or handle this case as needed
            example["image"] = np.zeros((self.size, self.size, 3), dtype=np.float32)
            return example

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


# ----------------------- #
# Dataset for the latents 
# ----------------------- #
class ProstateDatasetLatents(Dataset):
    def __init__(self,
                root,
                data_csv,
                model='sdxl-vae',
                size=256,
                normalization=1.,
                augmentation=False):
        
        base_path = f'{root}/{model}/{size}/'
        self.data_csv = data_csv

        self.image_ids = []
        self.gleason_list = []
        with open(self.data_csv, "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_ids.append(line.split(',')[0])
                self.gleason_list.append(line.split(',')[1])

        self.image_ids_set = set(self.image_ids) 

        if augmentation:
            base_path += 'aug/'
            image_dir = os.path.join(base_path, 'images')

            self.images = [
                os.path.join(image_dir, d+'-orig.npy')
                for d in self.image_ids]
    
        else:
            image_dir = os.path.join(base_path, 'images')

            #self.images = [
            #    os.path.join(image_dir, d)
            #    for d in sorted(os.listdir(image_dir))
            #    if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
            self.images = [
                os.path.join(image_dir, d+'.npy') for d in self.image_ids
            ]
                
        self.normalization = normalization

        image_ids_dict = {f"{image_id}.npy": index for index, image_id in enumerate(self.image_ids)}
        self.final_gleason = [int(self.gleason_list[image_ids_dict[image_name.split('/')[-1]]]) for image_name in tqdm(self.images)]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        r = rand(1)[0]
        if r <0.5:
            rep = 'orig'
        elif r<0.6:
            rep = 'vflip'
        elif r<0.7:
            rep = 'hflip'
        elif r<0.8:
            rep='rot90'
        elif r<0.9:
            rep = 'rot180'
        else:
            rep = 'rot270'

        img_path = self.images[idx].replace('orig', rep)
        
        example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization)),
                "gleason": self.final_gleason[idx]} 

        return example