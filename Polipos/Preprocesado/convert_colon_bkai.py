import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
import albumentations


def squarify_mask(mask):
    nz = np.nonzero(mask)
    if nz[0].shape != 0:
        x1 = np.min(nz[0])
        x2 = np.max(nz[0])
        y1 = np.min(nz[1])
        y2 = np.max(nz[1])
        square_mask = np.zeros(mask.shape, dtype=np.uint8)
        square_mask[x1:x2, y1:y2] = 1
        return square_mask
    else:
        return mask


train = False
size = 480
OPATH = f'/home/bejar/ssdstorage/bkai/'

if train:
    dataset = 'train'
else:
    dataset = 'test'

PATH = '/home/bejar/ssdstorage/bkai-igh-neopolyp/'
os.makedirs(f"{OPATH}/{dataset}/images", exist_ok=True)
images = sorted(glob(f"{PATH}/{dataset}/{dataset}/*.jpeg"))

if train:
    os.makedirs(f"{OPATH}/{dataset}/masks", exist_ok=True)
    masks = sorted(glob(f"{PATH}/{dataset}_gt/{dataset}_gt/*.jpeg"))


image_rescaler = albumentations.SmallestMaxSize(max_size=size,
                                                interpolation=cv2.INTER_CUBIC)
segmentation_rescaler = albumentations.SmallestMaxSize(
    max_size=size, interpolation=cv2.INTER_NEAREST)
preprocessor = albumentations.CenterCrop(height=size, width=size)


n = 0
for i in range(len(images)):
    print(images[i])
    image = np.array(Image.open(images[i])).astype(np.uint8)
    image = image_rescaler(image=image)["image"]

    if train:
        mask = np.array(Image.open(masks[i])).astype(np.uint8)
        mask = segmentation_rescaler(image=mask)["image"]

    if train:
        processed = preprocessor(image=image, mask=mask)
        mask = processed['mask']
        image = processed['image']

        if np.count_nonzero(mask) == 0:
            if len(mask.shape) == 3:
                sq_mask = mask[:, :, 0]
            else:
                sq_mask = mask
        else:
            if len(mask.shape) == 3:
                sq_mask = squarify_mask(processed['mask'][:, :, 0])
            else:
                sq_mask = squarify_mask(processed['mask'])

            sq_mask = np.array(sq_mask // np.max(sq_mask))

        Image.fromarray(image).save(f'{OPATH}/{dataset}/images/bkai-{i:04d}.jpg')
        np.save(f'{OPATH}/{dataset}/masks/bkai-{i:04d}.npy', sq_mask)
    else:
        processed = preprocessor(image=image)
        image = processed['image']
        Image.fromarray(image).save(f'{OPATH}/{dataset}/images/bkai-{i:04d}.jpg')
    n += 1
print(n)
