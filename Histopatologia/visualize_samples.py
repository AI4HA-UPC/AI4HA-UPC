import numpy as np
import os 
from PIL import Image

import torch 
from torch import nn 
from diffusion.models.autoencoder_kl import AutoencoderKL 


BASE_PATH = '/gpfs/projects/bsc70/bsc70174/PANDA_code/logs/PANDA-L-seg-DDPM-sdxlvae-F-s256-l32-124-r3-linear-t1000-lr5-wo_scaling-aug/samples'

samples_path = os.listdir(BASE_PATH)

images_paths = [s for s in samples_path if s.startswith('samples') and s.endswith('.npy')]
images_paths.sort()
masks_paths = [s for s in samples_path if s.startswith('masks') and s.endswith('.npy')]
masks_paths.sort()

images_latents = np.array([np.load(os.path.join(BASE_PATH, im), allow_pickle=True) for im in images_paths])
masks_latents = np.array([np.load(os.path.join(BASE_PATH, ma), allow_pickle=True) for ma in masks_paths])
images_latents_shape = images_latents.shape
masks_latents_shape = masks_latents.shape
print(images_latents_shape)
print(masks_latents_shape)
images_latents = images_latents.reshape(images_latents_shape[0]*images_latents_shape[1], images_latents_shape[2], images_latents_shape[3], images_latents_shape[4])
masks_latents = masks_latents.reshape(masks_latents_shape[0]*masks_latents_shape[1], masks_latents_shape[2], masks_latents_shape[3], masks_latents_shape[4])

AE = AutoencoderKL.from_pretrained('/gpfs/projects/bsc70/bsc70174/Models/sdxl-vae-fp16-fix').eval()
AE.requires_grad_(False)

decoded_images = []
masks = []
for i in range(images_latents.shape[0]):
    img = torch.from_numpy(np.expand_dims(images_latents[i], 0))
    decoded_image = AE.decode(img.permute(0, 3, 1, 2), return_dict=False)[0]
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = decoded_image.permute(0, 2, 3, 1).float().numpy()
    decoded_images.append(Image.fromarray((decoded_image[0]*255).astype('uint8')))

    mask = torch.from_numpy(np.expand_dims(np.moveaxis(masks_latents[i], -1, 0), 0))
    mask = nn.Upsample(scale_factor=8)(mask)
    mask = np.argmax(np.array(mask), axis=1)
    mask = torch.from_numpy(mask).permute(1, 2, 0).numpy()
    mask = np.repeat(mask, 3, axis=2)
    masks.append(Image.fromarray(((mask/5)*255).astype('uint8')))

decoded_images[0].save(f"{BASE_PATH}/samples.gif", save_all=True, append_images=decoded_images[1:], duration=600, loop=0)
masks[0].save(f"{BASE_PATH}/masks.gif", save_all=True, append_images=masks[1:], duration=600, loop=0)

print("Visualization completed!")

