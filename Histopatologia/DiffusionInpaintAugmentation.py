"""
Code to perform image inpainting as data augmentation technique. 
"""
import argparse
import logging
import os

from tqdm.auto import tqdm

from time import time
import numpy as np
import pandas as pd 

import torch
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, UNet2DConditionModel, PNDMPipeline, PNDMScheduler, DPMSolverMultistepScheduler
from diffusion.models.autoencoder_kl import AutoencoderKL

from diffusion.util import instantiate_from_config, load_config, time_management, print_memory, dataset_statistics, dataset_first_batch_std
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_pndm import PNDMPipeline
from diffusion.pipelines.pipeline_pndm_class import PNDMPipelineClass
from diffusion.pipelines.pipeline_pndm_inpainting import PNDMInpaintingPipeline
from diffusion.normalization.reinhard import Normalizer
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
    'PNDMInpaint': PNDMInpaintingPipeline
}

LATENTS = {
    "AEKL" : AutoencoderKL
}

MODEL = {
    "UNet2D": UNet2DModel,
    "UNet2DCondition": UNet2DConditionModel
}


####
# Function for data augmentation with inpainting
#### 
def inpaint_data_augmentation(
    original_image: torch.Tensor,
    original_mask: torch.Tensor,
    label: int,
    guidance_scale: float,
    num_inference_steps: int,
    pipeline: torch.nn.Module,
    AE: torch.nn.Module,
    normalizer: torch.nn.Module,
    num_variations: int = 4
):

    # Identify the values of interest
    label2gleason = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [3, 4],
        5: [4, 3],
        6: [4],
        7: [3, 5],
        8: [5, 3],
        9: [4, 5],
        10: [5, 4],
        11: [5]
    }
    values = label2gleason[label]

    # Process the image and define the binary mask
    #img_tensor = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0)
    img_tensor = original_image
    img_tensor = torch.cat([img_tensor]*num_variations, dim=0)
    img_tensor = (img_tensor - 0.5) * 2

    binary_mask = np.ones_like(original_mask)
    for v in values:
        binary_mask[original_mask == v] = 0
    mask_tensor = torch.from_numpy(binary_mask)#.unsqueeze(0).unsqueeze(0)
    #mask_tensor = torch.cat([mask_tensor, mask_tensor, mask_tensor], dim=1)
    mask_tensor = torch.cat([mask_tensor]*num_variations, dim=0)

    # Check the area of inpaint and filter
    area_inpaint = np.sum(binary_mask)/binary_mask.size
    print(f'Area inpaint: {area_inpaint}')
    if area_inpaint < 0.2 or area_inpaint > 0.7:
        print('Area inpaint too small')
        return None

    # Generate the variations
    generator = torch.Generator(device='cuda')
    final_image_latent = pipeline(
        initial_image=img_tensor.to('cuda'),
        mask=mask_tensor.to('cuda'),
        generator=generator,
        batch_size=num_variations,
        class_cond=torch.from_numpy(np.array([label]*num_variations)),
        embedding_type="onehot",
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps,
        output_type="numpy"
    ).images 

    norm_inv = T.Normalize(-1.0, 1.0)
    final_image = torch.from_numpy(final_image_latent)
    final_image = norm_inv(final_image.permute(0, 3, 1, 2).to('cuda'))
    final_image = AE.decode(final_image, return_dict=False)[0].cpu()
    final_image = (final_image / 2 + 0.5).clamp(0, 1).cpu()

    final_image_norm = []
    for i in range(num_variations):
        final_image_norm.append(torch.from_numpy(normalizer.transform(final_image[i].permute(1, 2, 0).numpy())))
    final_image_norm = torch.stack(final_image_norm).permute(0, 3, 1, 2)

    return final_image_norm


####
# Main 
####
def main(config):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()    
    BASE_DIR = f"{config['exp_dir']}/logs/{config['model_name']}" # logs
    SAVE_DIR = f"{config['out_dir']}/{config['dataset_name']}"

    OUT_IMAGES_DIR = f"{SAVE_DIR}/patches"

    os.makedirs(f"{SAVE_DIR}", exist_ok=True)
    os.makedirs(f"{OUT_IMAGES_DIR}", exist_ok=True)

    BATCH_SIZE = config['dataset']['dataloader']['batch_size']

    # Load the model 
    logger.info('LOADING MODEL')

    image_key = config["model"]["image_key"]

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    pipeline = PIPELINES[config['diffuser']['pipeline']].from_pretrained(f"{BASE_DIR}/model").to("cuda")
 
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
    labels = config['samples']['labels_to_generate']
    logger.info(f'Labels to be generated: {labels}') 

    label2gleason = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [3, 4],
        5: [4, 3],
        6: [4],
        7: [3, 5],
        8: [5, 3],
        9: [4, 5],
        10: [5, 4],
        11: [5]
    }

    # Original dataset
    train_data = instantiate_from_config(config['dataset']['train'])
    
    desired_dataset_size = np.round(len(train_data) // 1)
    random_indices = torch.randperm(len(train_data))[:desired_dataset_size]
    reduced_dataset = torch.utils.data.Subset(train_data, random_indices)

    train_dataloader = torch.utils.data.DataLoader(reduced_dataset, **config['dataset']["dataloader"])

    logger.info(f'Initial dataset size: {len(train_data)}') 

    # Normalizer 
    normalizer = Normalizer()
    normalizer.fit(image_path='/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/normalization/reference_img.png') # path

    # Augment the data
    global_step = 0
    file_names = []
    labels_generated = []
    for i, batch in enumerate(tqdm(train_dataloader)):
        image = batch[image_key]
        mask = batch['mask']
        label = batch['gleason']
        image_name = batch['image_name'][0]

        image = image.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        mask = torch.cat([mask, mask, mask], dim=1)
        label = [int(l) for l in label][0]

        if label not in labels:
            grid = T.ToPILImage(mode='RGB')(make_grid(image, nrow=1))
            grid.save(f"{OUT_IMAGES_DIR}/{image_name}_orig.png")

            file_names.append(f"{image_name}_orig")
            labels_generated.append(label)
            
            global_step += 1
            continue

        final_image_norm = inpaint_data_augmentation(
            original_image=image, 
            original_mask=mask, 
            label=label, 
            guidance_scale=config['samples']['guidance_scale'], 
            num_inference_steps=config['diffuser']['num_inference_steps'], 
            pipeline=pipeline,
            AE=AE,
            normalizer=normalizer,
            num_variations=config['samples']['num_variations']
        )

        if final_image_norm is None:
            grid = T.ToPILImage(mode='RGB')(make_grid(image, nrow=1))
            grid.save(f"{OUT_IMAGES_DIR}/{image_name}_orig.png")

            file_names.append(f"{image_name}_orig")
            labels_generated.append(label)
        else:
            grid = T.ToPILImage(mode='RGB')(make_grid(image, nrow=1))
            grid.save(f"{OUT_IMAGES_DIR}/{image_name}_orig.png")

            file_names.append(f"{image_name}_orig")
            labels_generated.append(label)

            for j in range(final_image_norm.size(0)):
                grid = T.ToPILImage(mode='RGB')(make_grid(final_image_norm[j, ...], nrow=1))
                grid.save(f"{OUT_IMAGES_DIR}/{image_name}_variation_{j+1}.png")

                file_names.append(f"{image_name}_variation_{j+1}")
                labels_generated.append(label)

        global_step += 1

        # Check the time
        last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break 

    csv_name = config['dataset']['train']['params']['data_csv'].split('/')[-1]
    new_df = pd.DataFrame(file_names, columns=["Image_id"])
    new_df['Gleason'] = labels_generated
    new_df.to_csv(os.path.join(SAVE_DIR, csv_name), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
    
    