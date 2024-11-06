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


PATH = '/home/bejar/ssdstorage/PolypGen2021_MultiCenterData_v3/'
OPATH = '/home/bejar/ssdstorage/MultiCenterData640/'
dataset = 'C5'

os.makedirs(f"{OPATH}/images", exist_ok=True)
os.makedirs(f"{OPATH}/masks", exist_ok=True)

images = sorted(glob(f"{PATH}/data_{dataset}/images_{dataset}/*.jpg"))
# masks = glob(f"{PATH}/masks/*.jpg")

size = 640
image_rescaler = albumentations.SmallestMaxSize(max_size=size,
                                                interpolation=cv2.INTER_CUBIC)
segmentation_rescaler = albumentations.SmallestMaxSize(
    max_size=size, interpolation=cv2.INTER_NEAREST)
preprocessor = albumentations.CenterCrop(height=size, width=size)


n = 0
for i in range(len(images)):
    print(images[i])
    image = np.array(Image.open(images[i])).astype(np.uint8)
    name = (images[i].split('.')[0] + '_mask.jpg').replace('images', 'masks')

    if image.shape[0] < size or image.shape[1] < size:
        continue

    mask = np.array(Image.open(name)).astype(np.uint8)

    if dataset == 'C4':
        image = image[:, 900:, :]
        mask = mask[:, 900:]
    if dataset == 'C5':
        image = image[:, 650:, :]
        mask = mask[:, 650:]
    image = image_rescaler(image=image)["image"]
    mask = segmentation_rescaler(image=mask)["image"]

    processed = preprocessor(image=image, mask=mask)
    image = processed['image']
    mask = processed['mask']
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
    Image.fromarray(image).save(
        f'{OPATH}/images/multicenter-{dataset}-{i:04d}.jpg')
    np.save(f'{OPATH}/masks/multicenter-{dataset}-{i:04d}.npy', sq_mask)
    n += 1
print(n)
