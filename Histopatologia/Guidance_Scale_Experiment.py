"""
Guidance Scale Experiment 
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision.transforms as T
import pandas as pd
import os
import random
from tqdm import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from accelerate import Accelerator
from sklearn.preprocessing import OneHotEncoder
from torchvision.utils import make_grid
from diffusion.util import load_config

from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_pndm import PNDMPipeline
from diffusion.pipelines.pipeline_pndm_class import PNDMPipelineClass
from diffusion.pipelines.pipeline_prompt import DPMSolverMask
from diffusion.pipelines.pipeline_ddim_class import DDIMPipelineClass
from diffusion.models.autoencoder_kl import AutoencoderKL
from diffusion.models.unet_2d_condition import UNet2DConditionModel


SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler,
    'DPMSolver': DPMSolverMultistepScheduler,
    'PNDM': PNDMScheduler
}

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline,
    'DPMSolver': DDPMPipeline,
    'PNDM': PNDMPipeline,
    'PNDMClass': PNDMPipelineClass,
    'DDIMClass': DDIMPipelineClass
}

id2label = {
    0: 'background/unknown',
    1: 'stroma',
    2: 'healthy epithelium',
    3: 'Gleason 3+3',
    4: 'Gleason 3+4',
    5: 'Gleason 4+3',
    6: 'Gleason 4+4',
    7: 'Gleason 3+5',
    8: 'Gleason 5+3',
    9: 'Gleason 4+5',
    10: 'Gleason 5+4',
    11: 'Gleason 5+5'
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

    out_dir = f"{config['base_dir']}/logs/{config['model_name']}"
    model_dir = f"{config['base_dir']}/logs/{config['model_name']}/model"

    # Import models
    unet = UNet2DConditionModel.from_pretrained(f"{model_dir}/unet")
    scheduler = SCHEDULERS[config['diffuser']['type']].from_pretrained(f"{model_dir}/scheduler")

    scheduler.config.clip_sample = False

    pipeline = PIPELINES[config['diffuser']['pipeline']](
            unet=unet,
            scheduler=scheduler
    ).to('cuda')

    AE = AutoencoderKL().from_pretrained(config['vae_dir']).eval()
    AE.to('cuda').requires_grad_(False)

    # One-hot encoding
    class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    class_labels = class_labels.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(class_labels)

    norm_inv = T.Normalize(-1.0, 1.0)

    # Guidance Scale Experiment 
    for g in config['experiment']['guidance_values']:
        print(f"Guidance: {g}")
        new_out_dir = os.path.join(out_dir, f'samples-g{g}')
        os.makedirs(new_out_dir, exist_ok=True)

        for c in tqdm(range(12)):
            print(f"Class: {id2label[c]}")

            for i in tqdm(range(int(config['experiment']['num_per_class']/config['experiment']['batch_size']))):
                seed = int(random.randint(0, 10000))
                generator = torch.Generator(device='cuda').manual_seed(seed)

                images = pipeline(
                    generator=generator,
                    batch_size=config['experiment']['batch_size'],
                    class_cond=torch.from_numpy(np.array([c]*config['experiment']['batch_size'])),
                    embedding_type="onehot",
                    guidance_scale=g,
                    num_inference_steps=config['diffuser']['num_inference_steps'],
                    output_type="numpy"
                ).images  

                images_chann = torch.from_numpy(images)
                latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                images_final = AE.decode(latents, return_dict=False)[0] 
                images_final = (images_final / 2 + 0.5).clamp(0, 1).cpu()

                for im in range(images_final.shape[0]):
                    grid = T.ToPILImage(mode='RGB')(make_grid(images_final[im].unsqueeze(0), nrow=1))
                    grid.save(f"{new_out_dir}/image_{seed:06d}_{g:02d}_{c:02d}_{(i*config['experiment']['batch_size']):04d}_{im:04d}.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)