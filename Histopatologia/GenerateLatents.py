import argparse
import logging
import os
from torchvision.transforms.functional import vflip, hflip
import torch
import torch.nn.functional as F
from skimage.measure import block_reduce
import time
import numpy as np
from diffusion.util import instantiate_from_config, load_config, time_management, print_memory
from tqdm import tqdm 


DIRS = ['samples']

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
        return  hflip(images), hflip(masks)
    if aug == 'rot90':
        return torch.rot90(images, k=1, dims=[2,3]),torch.rot90(masks, k=1, dims=[2,3])
    if aug == 'rot180':
        return torch.rot90(images, k=2, dims=[2,3]),torch.rot90(masks, k=2, dims=[2,3])
    if aug == 'rot270':
        return torch.rot90(images, k=3, dims=[2,3]),torch.rot90(masks, k=3, dims=[2,3])
    else:
        raise NameError('Augmentation not valid')


def main(config):

    if 'lib' in config['latents']:
        if config['latents']['lib'] == "diffusers":
           from diffusers.models.autoencoder_kl import AutoencoderKL
        else:
            from diffusion.models.autoencoder_kl import AutoencoderKL 
    else:
        from diffusion.models.autoencoder_kl import AutoencoderKL 

    LATENTS = {
        "AEKL" : AutoencoderKL
    }

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    OUT_DIR = config['out_dir'] 

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True) 

    logger = create_logger(BASE_DIR)
    image_key = 'image' if "image_key" not in config['dataset'] else config['dataset']['image_key']

    # Load the pretrained VAEKL or VQVAE
    latent_name = config['latents']['name']
    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    train_data = instantiate_from_config(config['dataset']['train'])
    test_data = instantiate_from_config(config['dataset']['test'])
    train_dataloader = torch.utils.data.DataLoader(train_data, **config['dataset']["dataloader"])

    logger.info(f"Dataset size: {len(train_data)}")
    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    img_size = config['dataset']['train']['params']['size']
    factor = config['dataset']['train']['params']['size'] // config['model']['params']['sample_size']
    logger.info(f"Factor: {factor}")

    augmentation = False if 'data_augmentation' not in config['dataset'] else config['dataset']['data_augmentation']

    FINAL_PATH = f"{OUT_DIR}/{latent_name}/{img_size}/"
    if augmentation:
        IMAGES_PATH = FINAL_PATH + 'aug/images'
        MASKS_PATH = FINAL_PATH + 'aug/masks'
    else:
        IMAGES_PATH = FINAL_PATH + 'images'
        MASKS_PATH = FINAL_PATH + 'masks'
    
    os.makedirs(IMAGES_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch[image_key].permute(0, 3, 1, 2)
        masks = batch['segmentation'].permute(0, 3, 1, 2)
        image_names = batch['image_name'] 

        if augmentation:
            transformations = ['orig', 'vflip', 'hflip', 'rot90', 'rot180', 'rot270']
        else:
            transformations = ['orig']
        
        for aug in transformations:
            clean_images, masks = apply_augmentation(clean_images, masks, aug)
            clean_images_latents = AE.encode(clean_images.to('cuda')).latent_dist.mode()

            for s in range(clean_images_latents.shape[0]):
                nsize = (1,factor,factor)
                nmask_1 = block_reduce(np.expand_dims(masks[s][0], axis=0), block_size=nsize, func=np.min)
                nmask_2 = block_reduce(np.expand_dims(masks[s][1], axis=0), block_size=nsize, func=np.min)
                nmask_3 = block_reduce(np.expand_dims(masks[s][2], axis=0), block_size=nsize, func=np.min)
                nmask_4 = block_reduce(np.expand_dims(masks[s][3], axis=0), block_size=nsize, func=np.min)
                nmask_5 = block_reduce(np.expand_dims(masks[s][4], axis=0), block_size=nsize, func=np.min)
                nmask_6 = block_reduce(np.expand_dims(masks[s][5], axis=0), block_size=nsize, func=np.min)
                
                nmask = np.concatenate((nmask_1, nmask_2, nmask_3, nmask_4, nmask_5, nmask_6), axis=0)

                if len(transformations) > 1:
                    np.save(f'{IMAGES_PATH}/{image_names[0]}_{step:08d}_{s:02d}-{aug}.npy', clean_images_latents[s].cpu().numpy())
                    np.save(f'{MASKS_PATH}/{image_names[0]}_{step:08d}_{s:02d}-{aug}_mask.npy', nmask)
                else:
                    np.save(f'{IMAGES_PATH}/{image_names[0]}_{step:08d}_{s:02d}.npy', clean_images_latents[s].cpu().numpy())
                    np.save(f'{MASKS_PATH}/{image_names[0]}_{step:08d}_{s:02d}_mask.npy', nmask)

        #logger.info(f'BATCH: {step}/{len(train_dataloader)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)


