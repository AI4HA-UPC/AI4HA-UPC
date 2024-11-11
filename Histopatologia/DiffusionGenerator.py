import argparse
import inspect
import logging
import math
import os

from tqdm.auto import tqdm

from time import time
import numpy as np
import shutil
import random 

import torch
from torch import nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, PNDMPipeline, PNDMScheduler, ScoreSdeVePipeline, ScoreSdeVeScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs
from diffusion.models.autoencoder_kl import AutoencoderKL

from diffusion.util import instantiate_from_config, load_config, time_management, print_memory, dataset_statistics
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_pndm import PNDMPipeline
from diffusion.pipelines.pipeline_pndm_class import PNDMPipelineClass
from diffusion.pipelines.pipeline_ddim_class import DDIMPipelineClass
from diffusion.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.transformers import DiTTransformer2DModel
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
    'PNDMClass': PNDMPipelineClass,
    'DDIMClass': DDIMPipelineClass
}

LATENTS = {
    "AEKL" : AutoencoderKL
}

MODEL = {
    "UNet2D": UNet2DModel,
    "UNet2DCondition": UNet2DConditionModel,
    "DiTTransformer2DModel": DiTTransformer2DModel
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
    diffuser = MODEL[config["model"]["modeltype"]].from_pretrained(f"{BASE_DIR}/model/unet")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        diffuser = nn.DataParallel(diffuser)
    
    if isinstance(diffuser, nn.DataParallel):
        original_diffuser = diffuser.module
    else:
        original_diffuser = diffuser
    
    diffuser.eval()
    diffuser.to('cuda')

    if config["model"]["modeltype"] == 'DiTTransformer2DModel':
        from diffusion.pipelines.pipeline_pndm_dit import PNDMPipeline

        PIPELINES = {
            'DDPM': DDPMPipeline,
            'DDIM': DDIMPipeline,
            'DPMSolver': DDPMPipeline,
            'PNDM': PNDMPipeline,
            'PNDMClass': PNDMPipelineClass,
            'DDIMClass': DDIMPipelineClass
        }

    image_key = config["model"]["image_key"]

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)
    
    if config['diffuser']['type'] == 'DDIM':
        noise_scheduler.config.clip_sample = False

    new_labels = config['model']['labels']

    #pipeline = PIPELINES[config['diffuser']['pipeline']].from_pretrained(f"{BASE_DIR}/model").to("cuda")

    #if not 'pipeline' in config['diffuser']:
    #    pipeline = PIPELINES[config['diffuser']['type']].from_pretrained(f"{BASE_DIR}/model").to("cuda")
        #pipeline = PIPELINES[config['diffuser']['type']](
        #    unet=diffuser,
        #    scheduler=noise_scheduler,
        #)
    #else:
    pipeline = PIPELINES[config['diffuser']['pipeline']](
        unet=diffuser,
        scheduler=noise_scheduler
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
    norm_inv = T.Normalize(-1.0, 1/xstd)

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = (config['time']) * 60
    else:
        time_budget = 2000

    for i in tqdm(range(config['nsamples'])):
        labels = range(len(new_labels))
        labels = random.choices(labels, k=config['batch'])
        labels = torch.tensor(labels)#.to('cuda')

        #labels = labels if 'class_embed_type' in config['model']["params"] else None

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
                output_type="numpy" # "pil"
            ).images 
        #logger.info(f'Latent shape: {images.shape}')
        #logger.info(f'Latent min-max: {images.min(), images.max()}')

        if image_key == 'image' and 'segmentation' not in config['model']:
            images = torch.from_numpy(images) 
            latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
            images = AE.decode(latents, return_dict=False)[0] 
            images = (images / 2 + 0.5).clamp(0, 1).cpu()
            #images = images.permute(0, 2, 3, 1)#.float().numpy()

            for im in range(images.shape[0]):
                grid = T.ToPILImage(mode='RGB')(make_grid(images[im], nrow=1))
                grid.save(f"{SAVE_DIR}/{seed:10d}_{(i*config['batch'])+im:04d}_sample.jpg")

        elif image_key == 'image' and config['model']['segmentation']:
            n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
            images_chann = torch.from_numpy(images[...,:n_channels])
            masks_chann = torch.from_numpy(images[...,n_channels:])

            latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
            images = AE.decode(latents, return_dict=False)[0] 
            images = (images / 2 + 0.5).clamp(0, 1).cpu()

            #logger.info(f'Latents shape: {latents.shape}')
            #logger.info(f'Latents min-max: {latents.min(), latents.max()}')
            #logger.info(f'Images shape: {images.shape}')
            #logger.info(f'Images min-max: {images.min(), images.max()}')
            
            mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
            mask = AE.decode(mask_latents, return_dict=False)[0] 
            mask = mask.clamp(0, 1).cpu()   
            for im in range(images.shape[0]):
                grid = T.ToPILImage(mode='RGB')(make_grid(images[im].unsqueeze(0), nrow=1))
                grid.save(f"{SAVE_DIR}/{seed:10d}_{(i*config['batch'] )+im:04d}_sample.jpg")
                grid = T.ToPILImage(mode='RGB')(make_grid(mask[im].unsqueeze(0), nrow=1))
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