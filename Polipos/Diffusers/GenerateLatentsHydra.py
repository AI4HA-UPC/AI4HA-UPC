################################################################################
# Script for transforming a dataset into latents using one of the SD VAEs      #
# using diffusers library and Hydra.                                           #
#                                                                              #
# The dataset is generated using different augmentations and the masks are     #
# downscaled accordingly to the downscaling of the VAE                         #
################################################################################
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
from torchvision.transforms.functional import vflip, hflip

import torch
from skimage.measure import block_reduce
from tqdm import tqdm
import time
import numpy as np
from ai4ha.util import instantiate_from_config, fix_paths
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


DIRS = ['samples', 'samples/images', 'samples/masks']


def create_logger(log_dir):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def apply_augmentation(images, masks, aug):
    if aug == 'orig':
        return images, masks
    if aug == 'vflip':
        return vflip(images), vflip(masks)
    if aug == 'hflip':
        return hflip(images), hflip(masks)
    if aug == 'rot90':
        return torch.rot90(images, k=1, dims=[2, 3]), torch.rot90(masks, k=1, dims=[2, 3])
    if aug == 'rot180':
        return torch.rot90(images, k=2, dims=[2, 3]), torch.rot90(masks, k=2, dims=[2, 3])
    if aug == 'rot270':
        return torch.rot90(images, k=3, dims=[2, 3]), torch.rot90(masks, k=3, dims=[2, 3])
    else:
        raise NameError('Augmentation not valid')


def main(config):

    if 'lib' in config['latents']:
        if config['latents']['lib'] == "diffusers":
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        else:
            from ai4ha.Autoencoders.autoencoder_kl import AutoencoderKL
    else:
        from ai4ha.Autoencoders.autoencoder_kl import AutoencoderKL

    LATENTS = {"AEKL": AutoencoderKL}

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    logger = create_logger(BASE_DIR)
    image_key = 'image' if "image_key" not in config['dataset'] else config[
        'dataset']['image_key']

    # Load the pretrained VAEKL or VQVAE
    AE = LATENTS[config['latents']['type']].from_pretrained(
        config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    train_data = instantiate_from_config(config['dataset']['train'])
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   **config["dataloader"])

    logger.info(f"Dataset size: {len(train_data)}")

    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")
    factor = config['dataset']['train']['params']['size'] // config['model'][
        'params']['sample_size']

    mask_channels = config['dataset']['train'][
        'mask_channels'] if 'mask_channels' in config['dataset']['train'] else 2
    logger.info(f"Factor: {factor} - Mask channels: {mask_channels}")

    # for step, batch in enumerate(train_dataloader):
    for step, batch in enumerate(tqdm(train_dataloader)):
        clean_images = batch[image_key].permute(0, 3, 1, 2)
        masks = batch['segmentation'].permute(0, 3, 1, 2)

        for aug in ['orig', 'vflip', 'hflip', 'rot90', 'rot180', 'rot270']:
            clean_images, masks = apply_augmentation(clean_images, masks, aug)
            clean_images_latents = AE.encode(
                clean_images.to('cuda')).latent_dist.mode()
            for s in range(clean_images_latents.shape[0]):
                # print(
                #     f'{BASE_DIR}/samples/images/image_{step:08d}_{s:02d}-{aug}.npy'
                # )
                np.save(
                    f'{BASE_DIR}/samples/images/image_{step:08d}_{s:02d}-{aug}.npy',
                    clean_images_latents[s].cpu().numpy())
                nsize = (1, factor, factor)
                if mask_channels == 2:
                    nmask_1 = block_reduce(np.expand_dims(masks[s][0], axis=0),
                                           block_size=nsize,
                                           func=np.max)
                    nmask_2 = block_reduce(np.expand_dims(masks[s][1], axis=0),
                                           block_size=nsize,
                                           func=np.min)
                    nmask = np.concatenate((nmask_1, nmask_2), axis=0)
                else:
                    nmask = block_reduce(np.expand_dims(masks[s][0], axis=0),
                                         block_size=nsize,
                                         func=np.max)
                np.save(
                    f'{BASE_DIR}/samples/masks/mask_{step:08d}_{s:02d}-{aug}.npy',
                    nmask)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def GenerateLatents(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    cfg['name'] = cfg['dataset']['name']
    main(cfg)


if __name__ == "__main__":
    GenerateLatents()
