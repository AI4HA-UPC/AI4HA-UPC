"""
Generation of images using conditional diffusion transformer model where the conditioning is the gleason score. 
"""
import argparse
import inspect
import logging
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DataParallel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import accelerate
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder

from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs
from transformers import AutoTokenizer, CLIPTextModel

from time import time
import numpy as np
import shutil
from diffusion.util import instantiate_from_config, load_config, time_management, print_memory, dataset_statistics, dataset_first_batch_std
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_pndm_dit import PNDMPipeline
from diffusion.pipelines.pipeline_pndm_class import PNDMPipelineClass
from diffusion.pipelines.pipeline_prompt import DPMSolverMask
from diffusion.models.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers import DiTTransformer2DModel
from torchvision.utils import make_grid
import torchvision.transforms as T

from packaging import version

# Definition of the schedulers and pipelines 
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
    'PNDM': PNDMPipeline
}

LATENTS = {
    "AEKL" : AutoencoderKL
}

DIRS = ['checkpoints','logs', 'samples', "final", "model"]

logger = get_logger(__name__, log_level="INFO")

    
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)

    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_diffuser_scheduler(config):
    scheduler = SCHEDULERS[config['diffuser']['type']]

    if 'clip_sample' not in config['diffuser']:
        params = {    
            'num_train_timesteps':config['diffuser']['num_steps'],
            'beta_schedule':config['diffuser']['beta_schedule']
        }
    else:
        clip = config['diffuser']['clip_sample']
        params = {    
            'num_train_timesteps':config['diffuser']['num_steps'],
            'beta_schedule':config['diffuser']['beta_schedule'],
            'clip_sample': clip
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

    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {num_gpus}')
    for i in range(num_gpus):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
    
    for i in range(num_gpus):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory Usage:')
        print(f'Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB')
        print(f'Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB')
    
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)   

    accparams = config['accelerator']
    # accparams["logging_dir"] = f"{BASE_DIR}/logs"
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(**config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if 'ema' in config:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if 'ema' in config:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), DiTTransformer2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DiTTransformer2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    print("Accelerator: ", accelerator.device)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    model = DiTTransformer2DModel(**config['model']['params'])

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    new_labels = {0: 'background/unknown',
                  1: 'stroma',
                  2: 'healthy epithelium',
                  3: '3+3',
                  4: '3+4',
                  5: '4+3',
                  6: '4+4',
                  7: '3+5',
                  8: '5+3',
                  9: '4+5',
                  10: '5+4',
                  11: '5+5'}

    # model.config_name=config['name']
    # model.disable_gradient_checkpointing()

    # Create EMA for the model.
    if 'ema' in config:
        ema_model = EMAModel(
            model.parameters(),
            decay=config['ema']['max_decay'],
            use_ema_warmup=True,
            inv_gamma=config['ema']['inv_gamma'],
            power=config['ema']['power'],
            model_cls=DiTTransformer2DModel,
            model_config=model.config,
        )

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    if config['diffuser']['type'] == 'DDIM':
        noise_scheduler.config.clip_sample = False

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'] * accelerator.num_processes,
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        weight_decay=config['optimizer']['weight_decay'],
        eps=config['optimizer']['epsilon'],
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    train_data = instantiate_from_config(config['dataset']['train'])
    test_data = instantiate_from_config(config['dataset']['test'])
    
    desired_dataset_size = np.round(len(train_data) // 1)
    random_indices = torch.randperm(len(train_data))[:desired_dataset_size]
    reduced_dataset = torch.utils.data.Subset(train_data, random_indices)

    train_dataloader = torch.utils.data.DataLoader(reduced_dataset, **config['dataset']["dataloader"])

    logger.info(f"Dataset size: {desired_dataset_size}")

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config['train']['lr_warmup_steps'] * accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) * config['train']['epochs']),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if 'ema' in config:
        ema_model.to(accelerator.device)

    total_batch_size = config['dataset']['dataloader']['batch_size'] * accelerator.num_processes * accparams['gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accparams['gradient_accumulation_steps'])
    max_train_steps = config['train']['epochs'] * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {desired_dataset_size}")
    logger.info(f"  Num Epochs = {config['train']['epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['dataset']['dataloader']['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Get the most recent checkpoint
    print(f'{BASE_DIR}/checkpoints/')
    dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    if dirs != []:
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = None

    logger.info(f'CHECKPOINT: {path}')

    resume_from_checkpoint = True
    if path is None:
        accelerator.print(
            f"Checkpoint does not exist. Starting a new training run."
        )
        resume_from_checkpoint = None
        resume_step = 0
    else:
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        global_step = int(path.split("_")[1]) *config['train']['checkpoint_freq']
        resume_global_step = global_step * accparams['gradient_accumulation_steps']
        first_epoch = (global_step // num_update_steps_per_epoch) 
        resume_step = (resume_global_step % (num_update_steps_per_epoch))# * accparams['gradient_accumulation_steps'])) 
        accelerator.print(f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}")

    hps = {"num_iterations": config['train']['epochs'], "learning_rate": config['train']['learning_rate'] * accelerator.num_processes}
    accelerator.init_trackers(config['name'][10:], config=hps, 
                              init_kwargs={"wandb":
                                           {"dir":os.path.join(BASE_DIR, "logs")}})
    
    #logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = config['time'] *60
    else:
        time_budget = 2000

    # Train!
    print_memory(logger, accelerator, "PRE LOOP")
    image_key ='image' if "image_key" not in config['model'] else config['model']['image_key']
    generator = torch.Generator(device=accelerator.device)
    start = 0

    # Compute dataset statistics when scaling
    if 'std_scaling' in config['train']:
        if 'segmentation' in config['model']:
            in_channels = config['model']['params']["in_channels"] - config['model']['segmentation']
        else:
            in_channels = config['model']['params']["in_channels"]
        if config['train']['std_scaling'] == "full":
            xmean, xstd = dataset_statistics(train_dataloader, image_key, channels=in_channels)
            #orm_tfm = T.Normalize(1.0, xstd) #xmean
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

    train_loss = []
    model.train()
    for epoch in range(first_epoch, config['train']['epochs']):
        # if start == 0:
        #     print_memory(logger, accelerator, "OUTER LOOP")

        noise_scheduler.set_timesteps(config['diffuser']['num_steps'])

        loss_epoch = 0

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
               
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % accparams['gradient_accumulation_steps'] == 0:
                    progress_bar.update(1)
                continue

            clean_images_latents = norm_tfm(batch[image_key])

            if "segmentation" in config['model']:
                if config['model']['segmentation'] > 0:
                    clean_images_latents = torch.cat((clean_images_latents, batch['segmentation']), 1)

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images_latents.shape).to(clean_images_latents.device)
            bsz = clean_images_latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images_latents.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images_latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # labels = None if 'class_ohe' not in batch else torch.squeeze(batch['class_ohe'])
                labels = None if 'gleason' not in batch else batch['gleason']
                
                model_output = model(noisy_images,
                                     timesteps,
                                     class_labels=labels.to('cuda')).sample
                
                #output_samples = [output.sample for output in model_output]
                #output_samples = [sample.to('cuda:1') for sample in output_samples]
                #model_output = torch.cat(output_samples, dim=0)

                # Different parameterization options (epsilon, clean sample or v) and different loss weighting options 
                if config['diffuser']['prediction_type'] == "epsilon":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod.to(clean_images_latents.device), timesteps, (clean_images_latents.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)

                    if config["train"]["weighting"] == "constant":
                        loss_weight = 1 / (snr_weights + 0.0001)
                    elif config["train"]["weighting"] == "snr":
                        loss_weight = 1
                    elif config["train"]["weighting"] == "max-snr-lambda":
                        loss_weight = torch.maximum(snr_weights, torch.tensor([1]).to('cuda')) / (snr_weights + 0.0001)
                    elif config["train"]["weighting"] == "min-snr-lambda":
                        loss_weight = torch.minimum(snr_weights, torch.tensor([5]).to('cuda')) / (snr_weights + 0.0001)
                    else:
                        raise ValueError("Invalid weighting!")
                    
                    loss = loss_weight * F.mse_loss(model_output, noise) 
                    loss = loss.mean()
                    loss_epoch += loss.detach().cpu()
                    
                elif config['diffuser']['prediction_type'] == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod.to(clean_images_latents.device), timesteps, (clean_images_latents.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)

                    if config["train"]["weighting"] == "constant":
                        loss_weight = 1
                    elif config["train"]["weighting"] == "snr":
                        loss_weight = snr_weights
                    elif config["train"]["weighting"] == "max-snr-lambda":  
                        loss_weight = torch.maximum(snr_weights, torch.tensor([1]).to('cuda'))
                    elif config["train"]["weighting"] == "min-snr-lambda":
                        loss_weight = torch.minimum(snr_weights, torch.tensor([5]).to('cuda'))
                    else:
                        raise ValueError("Invalid weighting!")

                    loss = loss_weight * F.mse_loss(model_output, clean_images_latents, reduction="none") 
                    loss = loss.mean()
                    loss_epoch += loss.detach().cpu()

                elif config['diffuser']['prediction_type'] == "v_prediction":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod.to(clean_images_latents.device), timesteps, (clean_images_latents.shape[0], 1, 1, 1)
                    )
                    v_pred = (alpha_t)**0.5 * noise - (1 - alpha_t)**0.5 * clean_images_latents
                    snr_weights = alpha_t / (1 - alpha_t)

                    if config["train"]["weighting"] == "constant":
                        loss_weight = 1 / (snr_weights + 1)
                    elif config["train"]["weighting"] == "snr":
                        loss_weight = snr_weights / (snr_weights + 1)
                    elif config["train"]["weighting"] == "max-snr-lambda":
                        loss_weight = torch.maximum(snr_weights, torch.tensor([1]).to('cuda')) / (snr_weights + 1)
                    elif config["train"]["weighting"] == "min-snr-lambda":
                        loss_weight = torch.minimum(snr_weights, torch.tensor([5]).to('cuda')) / (snr_weights + 1)
                    else:
                        raise ValueError("Invalid weighting!")

                    loss = loss_weight * F.mse_loss(model_output, v_pred, reduction="none")
                    loss = loss.mean()
                    loss_epoch += loss.detach().cpu()
                else:
                    raise ValueError(f"Unsupported prediction type: {config['diffuser']['prediction_type']}")

                accelerator.backward(loss, retain_graph=True)
                #accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if 'ema' in config:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if (global_step+1) % config['train']['checkpoint_freq'] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(f'{BASE_DIR}/checkpoints/', f"checkpoint_{global_step//config['train']['checkpoint_freq']:06d}") #
                        # save_path = f'{BASE_DIR}/checkpoints/'
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
                        dirs = sorted([d for d in dirs if d.startswith("checkpoint")])
                        if len(dirs) > config["projectconf"]["total_limit"]:
                            for d in dirs[:-config["projectconf"]["total_limit"]]:
                                logger.info(f'delete {BASE_DIR}/checkpoints/{d}')
                                shutil.rmtree(f'{BASE_DIR}/checkpoints/{d}')

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if 'ema' in config:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        train_loss.append(loss_epoch / len(train_dataloader))
        print("Train loss: ", train_loss)

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            accelerator.get_tracker("wandb").log({"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]},step=global_step)
            unet = accelerator.unwrap_model(model)
            if 'ema' in config:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())


            pipeline = PIPELINES[config['diffuser']['pipeline']](
                unet=unet.to('cuda'),
                scheduler=noise_scheduler
            ).to(accelerator.device)

            if epoch % config['samples']['samples_freq'] == 0 or epoch == config['train']['epochs'] - 1:
                # generator = torch.Generator(device=pipeline.device).seed() #.manual_seed(global_step)
                nsamp = 5 if 'samples_num' not in config['samples'] else config['samples']['samples_num']
                
                for g in range(nsamp):
                    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    labels = random.choices(labels, k=1)
                    labels = torch.tensor(labels)
                    
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        class_cond=labels,
                        batch_size=1, #config['dataset']['batch_size'],
                        num_inference_steps=config['diffuser']['num_inference_steps'],
                        guidance_scale=1,
                        output_type="numpy" # "pil"
                    ).images 
                    
                    # np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)
                    
                    if image_key == 'image' and 'segmentation' not in config['model']:
                        images = torch.from_numpy(images)
                        np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)

                        #images = torch.from_numpy(images) 
                        latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = (images / 2 + 0.5).clamp(0, 1).cpu()
                        #images = images.permute(0, 2, 3, 1)#.float().numpy()

                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                    elif image_key == 'segmentation' and 'segmentation' not in config['model']:
                        np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)

                        #images = torch.from_numpy(images) 
                        latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = images[:, 0, ...].unsqueeze(1) 
                        images = images.clamp(0, 1).cpu()
                        #images = images.permute(0, 2, 3, 1)#.float().numpy()
                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                    elif image_key == 'image' and config['model']['segmentation']:
                        n_channels = config['model']['params']['out_channels'] - config['model']['segmentation']
                        images_chann = torch.from_numpy(images[...,:n_channels])
                        masks_chann = torch.from_numpy(images[...,n_channels:])

                        np.save(f"{BASE_DIR}/samples/sample_{global_step:06d}-{g:02d}.npy", images_chann)
                        latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = (images / 2 + 0.5).clamp(0, 1).cpu()
                        #images = images.permute(0, 2, 3, 1).float().numpy()
                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                        np.save(f"{BASE_DIR}/samples/mask_{epoch:04d}-{global_step:06d}-{g:02d}.npy", masks_chann)
                        mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
                        mask = AE.decode(mask_latents, return_dict=False)[0] 
                        mask = mask[:, 0, ...].unsqueeze(1) 
                        mask = mask.clamp(0, 1).cpu()
                        grid = T.ToPILImage(mode='RGB')(make_grid(mask, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/mask_{global_step:06d}_{g:02d}.png")


            if 'ema' in config:
                ema_model.restore(unet.parameters())

            if (epoch % config['train']['checkpoint_epoch_freq'] == 0) or epoch == (config['train']['epochs'] - 1):
                # save the model
                if os.path.exists(f"{BASE_DIR}/model.last"):
                    shutil.rmtree(f'{BASE_DIR}/model.last')
                
                if os.path.exists(f"{BASE_DIR}/model"):
                    os.rename(f"{BASE_DIR}/model", f"{BASE_DIR}/model.last")

                unet = accelerator.unwrap_model(model)
                if 'ema' in config:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = PIPELINES[config['diffuser']['pipeline']](
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(f"{BASE_DIR}/model")

                if 'ema' in config:
                    ema_model.restore(unet.parameters())


        last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break        

    # Save loss 
    train_loss_floats = [float(tensor) for tensor in train_loss]
    df = pd.DataFrame(train_loss_floats, columns=['train_loss'])
    csv_file_path = f"{BASE_DIR}/logs/train_loss.csv"
    df.to_csv(csv_file_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(first_epoch, epoch+1)), train_loss, 'bo-', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{BASE_DIR}/train_val_loss.png')

    logger.info(f"Finish training epoch = {epoch}")
    # Generate images with the last model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        pipeline = PIPELINES[config['diffuser']['pipeline']](
            unet=unet,
            scheduler=noise_scheduler,
        )      

        # generator = torch.Generator(device=pipeline.device).seed()
        for g in range(config['samples']['samples_gen']):

            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            labels = random.choices(labels, k=1)
            labels = torch.tensor(labels)
            #labels = labels.to('cuda')
     
            images = pipeline(
                generator=generator,
                class_cond=labels,
                batch_size=1, #config['dataset']['batch_size'],
                num_inference_steps=config['diffuser']['num_inference_steps'],
                output_type="numpy" # "pil"
            ).images 

            # np.save(f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy", images)

            if image_key == 'image' and 'segmentation' not in config['model']:
                np.save(f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy", images)

                images = torch.from_numpy(images) 
                latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()
                #images = images.permute(0, 2, 3, 1)#.float().numpy()
                grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")

            elif image_key == 'segmentation' and 'segmentation' not in config['model']:
                    np.save(f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy", images)

                    images = torch.from_numpy(images) 
                    latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                    images = AE.decode(latents, return_dict=False)[0] 
                    images = images[:, 0, ...].unsqueeze(1) 
                    images = images.clamp(0, 1).cpu()
                    #images = images.permute(0, 2, 3, 1)#.float().numpy()
                    grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                    grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")

            elif image_key == 'image' and config['model']['segmentation']:
                n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
                images_chann = torch.from_numpy(images[...,:n_channels])
                masks_chann = torch.from_numpy(images[...,n_channels:])

                np.save(f"{BASE_DIR}/final/sample_{global_step:06d}-{g:02d}.npy", images_chann)

                latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()
                #images = images.permute(0, 2, 3, 1).float().numpy()
                grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")

                np.save(f"{BASE_DIR}/final/mask_{epoch:04d}-{global_step:06d}-{g:02d}.npy", masks_chann)
                mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
                mask = AE.decode(mask_latents, return_dict=False)[0] 
                mask = mask[:, 0, ...].unsqueeze(1) 
                mask = mask.clamp(0, 1).cpu()
                grid = T.ToPILImage(mode='RGB')(make_grid(mask, nrow=1))
                grid.save(f"{BASE_DIR}/final/mask_{global_step:06d}_{g:02d}.png")

    accelerator.end_training()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)