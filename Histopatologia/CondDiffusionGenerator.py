import argparse
import inspect
import logging
import math
import os

from tqdm.auto import tqdm

from time import time
import numpy as np
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, UNet2DConditionModel, PNDMPipeline, PNDMScheduler, ScoreSdeVePipeline, ScoreSdeVeScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs
from diffusion.models.autoencoder_kl import AutoencoderKL

from diffusion.util import instantiate_from_config, load_config, time_management, print_memory, dataset_statistics
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_ddim_image import DDIMImagePipeline
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
    'DDIMIMage': DDIMImagePipeline
}

LATENTS = {
    "AEKL" : AutoencoderKL
}

MODEL = {
    "UNet2D": UNet2DModel,
    "UNet2DCondition": UNet2DConditionModel
}


def get_diffuser_scheduler(config):
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

    return scheduler(**params)


def main(config):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()    
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    SAVE_DIR = f"{BASE_DIR}/gsamples/{config['diffuser']['num_inference_steps']}"

    os.makedirs(f"{SAVE_DIR}", exist_ok=True)

    logger.info('LOADING MODEL')
    diffuser = MODEL[config["model"]["modeltype"]].from_pretrained(f"{BASE_DIR}/model.last/unet")

    diffuser.eval()
    diffuser.to('cuda')

    image_key = config["model"]["image_key"]

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    # Dataset to extract the reference masks
    train_data = instantiate_from_config(config['dataset']['train'])
    train_dataloader = DataLoader(train_data, **config['dataset']["dataloader"])

    num_samples = config["nsamples"] #* config['batch']
    selected_dataset, remaining_dataset = random_split(train_data, [num_samples, len(train_data) - num_samples])

    # Convert the selected_dataset to a DataLoader if needed
    selected_data_loader = DataLoader(selected_dataset, batch_size=1, shuffle=True)

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    if not 'pipeline' in config['diffuser']:
        pipeline = PIPELINES[config['diffuser']['type']].from_pretrained(f"{BASE_DIR}/model").to("cuda")
        #pipeline = PIPELINES[config['diffuser']['type']](
        #    unet=diffuser,
        #    scheduler=noise_scheduler,
        #)
    else:
        pipeline = PIPELINES[config['diffuser']['pipeline']](
            unet=diffuser,
            scheduler=noise_scheduler,
        )    
    generator = torch.Generator(device='cuda')
    seed = generator.seed()

    # Compute dataset statistics when scaling
    if 'std_scaling' in config['train']:
        if 'segmentation' in config['model']:
            in_channels = config['model']['params']["in_channels"] - config['model']['segmentation']
        else:
            in_channels = config['model']['params']["in_channels"]
        if config['train']['std_scaling'] == "full":
            xmean, xstd = dataset_statistics(train_dataloader, image_key, channels=in_channels)
            #norm_tfm = T.Normalize(1.0, xstd) #xmean
        elif config['train']['std_scaling'] == "batch":
            xstd = dataset_first_batch_std(train_data, config['dataset']["dataloader"], image_key)
            #norm_tfm = T.Normalize(1.0, xstd)        
        elif 'std_scaling_val' in config['train']:
            xstd = config['train']['std_scaling_val']
            #norm_tfm = T.Normalize(1.0, xstd)
        else:
            xstd = 1 / 0.18215
            #norm_tfm = T.Normalize(1.0, xstd)
    else:
        xstd = 1
        #norm_tfm = T.Normalize(1.0, xstd)
    norm_tfm = T.Normalize(1.0, xstd)
    norm_inv = T.Normalize(1.0, 1/xstd)

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = (config['time']) * 60
    else:
        time_budget = 2000

    for i in range(config['nsamples']):
        # Extract reference mask 
        reduced_mask_latents = next(iter(selected_data_loader))["segmentation_reduced"]
        reduced_mask_latents = reduced_mask_latents.repeat(config['batch'], 1, 1, 1)

        mask_latents = next(iter(selected_data_loader))["segmentation"]

        images = pipeline(
            generator=generator,
            image_cond_latent=reduced_mask_latents.to('cuda'),
            guidance_scale = 7.5,
            batch_size=config['batch'],
            num_inference_steps=config['diffuser']['num_inference_steps'],
            output_type="numpy"
        ).images 

        if image_key == 'image' and 'segmentation' not in config['model']:
            images = torch.from_numpy(images) 
            latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
            images = AE.decode(latents, return_dict=False)[0] 
            images = (images / 2 + 0.5).clamp(0, 1).cpu()
            #images = images.permute(0, 2, 3, 1)#.float().numpy()

            reference_mask = AE.decode(mask_latents.to('cuda'), return_dict=False)[0] 
            reference_mask = reference_mask[:, 0, ...].unsqueeze(1)
            reference_mask = reference_mask.clamp(0, 1).cpu().numpy()
            reference_mask = np.round(reference_mask)
            reference_mask = torch.from_numpy(reference_mask)
            grid = T.ToPILImage(mode='RGB')(make_grid(reference_mask/5, nrow=1))
            grid.save(f"{SAVE_DIR}/reference_mask_{i:04d}.jpg")

            for im in range(images.shape[0]):
                grid = T.ToPILImage(mode='RGB')(make_grid(images[im], nrow=1))
                grid.save(f"{SAVE_DIR}/reference_mask_{i:04d}_{im:02d}_sample.jpg")

        elif image_key == 'image' and config['model']['segmentation']:
            n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
            images_chann = torch.from_numpy(images[...,:n_channels])
            masks_chann = torch.from_numpy(images[...,n_channels:])

            latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
            images = AE.decode(latents, return_dict=False)[0] 
            images = (images / 2 + 0.5).clamp(0, 1).cpu()
            
            mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
            mask = AE.decode(mask_latents, return_dict=False)[0] 
            mask = mask[:, 0, ...].unsqueeze(1)
            mask = mask.clamp(0, 1).cpu()
            for im in range(images.shape[0]):
                grid = T.ToPILImage(mode='RGB')(make_grid(images[im], nrow=1))
                grid.save(f"{SAVE_DIR}/{seed:10d}_{(i*config['batch'] )+im:04d}_sample.jpg")
                grid = T.ToPILImage(mode='RGB')(make_grid(mask[im], nrow=1))
                grid.save(f"{SAVE_DIR}/{seed:10d}_{(i*config['batch'] )+im:04d}_mask.png")

        last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)