import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F


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


PATH = '/home/bejar/ssdstorage/Colon/Kvasir-SEG/'
OPATH = '/home/bejar/ssdstorage/Kvasir-SEG-C/'

images = glob(f"{PATH}/images/*.jpg")
masks = glob(f"{PATH}/masks/*.jpg")

size = 480
hsize = 550
vsize = 650
n = 0
for i in range(len(images)):
    image = Image.open(images[i])
    h, w, c = np.asarray(image).shape
    if h < size or w < size:
        continue
    if h > hsize or w > vsize:
        continue
    print(h, w, i, '<--')
    cropped_img = F.center_crop(image, size)
    image = np.asarray(cropped_img)
    mask = Image.open(masks[i])
    cropped_mask = F.center_crop(mask, size)
    mask = np.asarray(cropped_mask)
    sq_mask = squarify_mask(mask[:, :, 0]).astype(np.uint8)
    sq_mask = np.array(sq_mask // np.max(sq_mask))
    Image.fromarray(image).save(f'{OPATH}/images/kvasir-seg-{i:03d}.jpg')
    np.save(f'{OPATH}/masks/kvasir-seg-{i:03d}.npy', sq_mask)
    n += 1
print(n)
