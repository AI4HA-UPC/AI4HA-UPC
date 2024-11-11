"""
Code to generate a synthetic dataset. 
"""
import argparse
import inspect
import logging
import os

from tqdm.auto import tqdm

from time import time
import numpy as np
import pandas as pd 

import torch
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, PNDMPipeline, PNDMScheduler, DPMSolverMultistepScheduler
from diffusion.models.autoencoder_kl import AutoencoderKL

from diffusion.util import load_config, time_management
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_pndm import PNDMPipeline
from diffusion.pipelines.pipeline_pndm_class import PNDMPipelineClass
from diffusion.models.unet_2d_condition import UNet2DConditionModel
from torchvision.utils import make_grid
import torchvision.transforms as T


SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler,
    'PNDM': PNDMScheduler,
    'DPMSolver': DPMSolverMultistepScheduler
}

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline,
    'DPMSolver': DDPMPipeline,
    'PNDM': PNDMPipeline,
    'PNDMClass': PNDMPipelineClass
}

LATENTS = {
    "AEKL" : AutoencoderKL
}

MODEL = {
    "UNet2D": UNet2DModel,
    "UNet2DCondition": UNet2DConditionModel
}

def get_diffuser_scheduler(config):
    """
    Get the scheduler and its configuration.

    :param config: dictionary with the configuration of the scheduler.
    """
    scheduler = SCHEDULERS[config['diffuser']['type']]

    params = {    
        'num_train_timesteps':config['diffuser']['num_steps'],
        'beta_schedule':config['diffuser']['beta_schedule']
    }

    if "prediction_type" in set(inspect.signature(scheduler.__init__).parameters.keys()):
        params['prediction_type']=config['diffuser']['prediction_type']
    if ("variance_type" in set(inspect.signature(scheduler.__init__).parameters.keys())) and  ("variance_type" in config['diffuser']):
        params['variance_type']=config['diffuser']['variance_type']
    if "betas" in config['diffuser']:
        params['beta_start']=config['diffuser']['betas'][0]
        params['beta_end']=config['diffuser']['betas'][1] 
    if "solver_order" in set(inspect.signature(scheduler.__init__).parameters.keys()):
        params['solver_order'] = config['diffuser']['solver_order']
    if "rescale_betas_zero_snr" in set(inspect.signature(scheduler.__init__).parameters.keys()):
        params['rescale_betas_zero_snr'] = config['diffuser']['rescale_betas_zero_snr'] 

    return scheduler(**params)


def main(config):
    """
    Main function.

    :param config: configuration file.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()    
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    SAVE_DIR = f"{config['out_dir']}/{config['name']}"

    OUT_IMAGES_DIR = f"{SAVE_DIR}/patches"
    OUT_MASKS_DIR = f"{SAVE_DIR}/masks"
    OUT_LATENTS_DIR = f"{SAVE_DIR}/latents"

    os.makedirs(f"{SAVE_DIR}", exist_ok=True)
    os.makedirs(f"{OUT_IMAGES_DIR}", exist_ok=True)
    os.makedirs(f"{OUT_MASKS_DIR}", exist_ok=True)
    os.makedirs(f"{OUT_LATENTS_DIR}", exist_ok=True)

    # Load the model 
    logger.info('LOADING MODEL')
    diffuser = MODEL[config["model"]["modeltype"]].from_pretrained(f"{BASE_DIR}/model/unet")
    
    diffuser.eval()
    diffuser.to('cuda')

    image_key = config["model"]["image_key"]

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    if config['diffuser']['type'] == 'DDIM':
        noise_scheduler.config.clip_sample = False

    pipeline = PIPELINES[config['diffuser']['pipeline']](
        unet=diffuser,
        scheduler=noise_scheduler
    )    
 
    generator = torch.Generator(device='cuda')
    seed = generator.seed()

    # Define normalization when scaling
    if 'std_scaling' in config['train']:
        xstd = 1 / 0.18215
    else:
        xstd = 1
    norm_tfm = T.Normalize(1.0, xstd)
    norm_inv = T.Normalize(-1.0, 1/xstd)

    # Set the time budget
    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = (config['time']) * 60
    else:
        time_budget = 2000

    # Labels to generate 
    samples_per_label = config['nsamples'] / 12
    samples_per_class = config['nsamples'] / 8
    logger.info(f'Samples to be generated in total: {config["nsamples"]*config["batch"]}')
    logger.info(f'Samples to be generated per label: {samples_per_label*config["batch"]}')

    dict_samples = {
        '0': samples_per_class,
        '1': samples_per_class,
        '2': samples_per_class,
        '3': samples_per_class,
        '4': samples_per_class,
        '5': samples_per_class,
        '6': samples_per_class/3,
        '7': samples_per_class/3,
        '8': samples_per_class/3,
        '9': samples_per_class/3,
        '10': samples_per_class/3,
        '11': samples_per_class/3
    }

    file_names = []
    labels_generated = []

    for c in range(12):
        for i in tqdm(range(int(dict_samples[str(c)]))):
            label = c
            labels = torch.tensor([label]*config['batch']).to(torch.int64)#.to('cuda')
            labels_generated.extend(labels.numpy())

            labels = labels if 'class_embed_type' in config['model']["params"] else None

            logger.info(f'Labels of the batch: {labels}')

            if 'guidance_scale' in config["samples"]:
                images = pipeline(
                    generator=generator,
                    class_cond=labels,
                    embedding_type=config['model']['embeddings'],
                    guidance_scale=config["samples"]["guidance_scale"],
                    batch_size=config['batch'],
                    num_inference_steps=config['diffuser']['num_inference_steps'],
                    output_type="numpy" 
            ).images 
            else:
                images = pipeline(
                    generator=generator,
                    class_cond=labels,
                    batch_size=config['batch'],
                    num_inference_steps=config['diffuser']['num_inference_steps'],
                    output_type="numpy"
                ).images 

            if image_key == 'image' and 'segmentation' not in config['model']:
                images = torch.from_numpy(images) 
                latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()

                for im in range(images.shape[0]):
                    grid = T.ToPILImage(mode='RGB')(make_grid(images[im], nrow=1))
                    grid.save(f"{OUT_IMAGES_DIR}/image_{seed:10d}_{c:04d}_{(i*config['batch'])}_{im:04d}.jpg")

                    np.save(f"{OUT_LATENTS_DIR}/image_{seed:10d}_{c:04d}_{(i*config['batch'])}_{im:04d}.npy", latents[im].cpu().numpy())

                    file_names.append(f'image_{seed:10d}_{c:04d}_{(i*config["batch"])}_{im:04d}')

            elif image_key == 'image' and config['model']['segmentation']:
                n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
                images_chann = torch.from_numpy(images[...,:n_channels])
                masks_chann = torch.from_numpy(images[...,n_channels:])

                latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()
                
                mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
                mask = AE.decode(mask_latents, return_dict=False)[0] 
                mask = torch.mean(mask, dim=1).unsqueeze(1)
                mask = mask.clamp(0, 1).cpu()
                mask = mask * 5
                mask = torch.round(mask)
                mask = mask.to(torch.uint8)
                for im in range(images.shape[0]):
                    grid = T.ToPILImage(mode='RGB')(make_grid(images[im].unsqueeze(0), nrow=1))
                    grid.save(f"{OUT_IMAGES_DIR}/image_{seed:10d}_{c:04d}_{(i*config['batch']):04d}_{im:04d}.png")
                    grid = T.ToPILImage(mode='RGB')(make_grid(mask[im].unsqueeze(0), nrow=1))
                    grid.save(f"{OUT_MASKS_DIR}/image_{seed:10d}_{c:04d}_{(i*config['batch']):04d}_{im:04d}_mask.png")

                    file_names.append(f'image_{seed:10d}_{c:04d}_{(i*config["batch"]):04d}_{im:04d}')

            last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
            if (time_budget < np.mean(qe_time)):
                break 

    new_df = pd.DataFrame(file_names, columns=["Image_id"])
    new_df['Gleason'] = labels_generated
    new_df.to_csv(os.path.join(SAVE_DIR, f"generated_data_{seed:10d}.csv"), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)