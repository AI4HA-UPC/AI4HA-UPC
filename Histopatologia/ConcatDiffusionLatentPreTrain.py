import argparse
import inspect
import logging
import math
import os

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import accelerate
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs

from time import time
import numpy as np
import shutil
from diffusion.util import instantiate_from_config, load_config, time_management, print_memory, dataset_statistics, dataset_first_batch_std
from diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from diffusion.pipelines.pipeline_ddim import DDIMPipeline
from diffusion.pipelines.pipeline_ddim_image import DDIMImagePipeline
from diffusion.pipelines.pipeline_concat_image import DPMSolverConcat
from diffusion.models.autoencoder_kl import AutoencoderKL
from torchvision.utils import make_grid
import torchvision.transforms as T

from packaging import version

SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler,
    'DPMSolver': DPMSolverMultistepScheduler,
}

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline,
    'DDIMIMage': DDIMImagePipeline,
    'DPMSolver': DDPMPipeline,
    'DPMSolverConcat': DPMSolverConcat
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
    scheduler = SCHEDULERS[config['diffuser']['scheduler_type']]

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

    return scheduler(**params)



def main(config):
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
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    model = UNet2DConditionModel(**config['model']['params'])

    AE = LATENTS[config['latents']['type']].from_pretrained(config['latents']['model']).eval()
    AE.to('cuda').requires_grad_(False)

    clip_model = CLIPTextModel.from_pretrained('/gpfs/projects/bsc70/bsc70174/Models/stable-diffusion-2-1', subfolder='text_encoder')
    tokenizer = AutoTokenizer.from_pretrained('/gpfs/projects/bsc70/bsc70174/Models/stable-diffusion-2-1/tokenizer')

    inputs = tokenizer([" "]*config['dataset']['dataloader']['batch_size'], padding=True, return_tensors="pt")
    prompt_embeds = clip_model(inputs.input_ids, attention_mask=None)
    prompt_embeds = prompt_embeds[0]

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
            model_cls=UNet2DConditionModel,
            model_config=model.config
        )

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

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

    desired_dataset_size = np.round(len(train_data) // 6)
    random_indices = torch.randperm(len(train_data))[:desired_dataset_size]
    reduced_dataset = torch.utils.data.Subset(train_data, random_indices)

    train_dataloader = torch.utils.data.DataLoader(reduced_dataset, **config['dataset']["dataloader"])

    logger.info(f"Dataset size: {len(train_data)}")

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
    logger.info(f"  Num examples = {len(train_data)}")
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
    print(dirs)
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
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = (resume_global_step % (num_update_steps_per_epoch))# * accparams['gradient_accumulation_steps'])) 
        accelerator.print(f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}")

    hps = {"num_iterations": config['train']['epochs'], "learning_rate": config['train']['learning_rate'] * accelerator.num_processes}
    accelerator.init_trackers(config['name'], config=hps, 
                              init_kwargs={"wandb":
                                           {"dir":os.path.join(BASE_DIR, "logs")}})
    
    # logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

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
        xstd = 1.0
        #norm_tfm = T.Normalize(1.0, xstd)
    norm_tfm = T.Normalize(1.0, xstd)
    norm_inv = T.Normalize(1.0, 1/xstd)


    #model.train()
    for epoch in range(first_epoch, config['train']['epochs']):
        model.train()
        # if start == 0:
        #     print_memory(logger, accelerator, "OUTER LOOP")

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # if start == 0:
            #     print_memory(logger, accelerator, "INNER LOOP")
               
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
                model_output = model(noisy_images,
                                     timesteps,
                                     encoder_hidden_states=prompt_embeds.to('cuda')).sample 

                if config['diffuser']['prediction_type'] == "epsilon":
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                elif config['diffuser']['prediction_type'] == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod.to(clean_images_latents.device), timesteps, (clean_images_latents.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(model_output, clean_images_latents, reduction="none")  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {config['diffuser']['prediction_type']}")

                accelerator.backward(loss, retain_graph=True)

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

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            accelerator.get_tracker("wandb").log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},step=global_step)
            unet = accelerator.unwrap_model(model)
            if 'ema' in config:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())


            pipeline = PIPELINES[config['diffuser']['pipeline_type']](
                unet=unet,
                scheduler=noise_scheduler
            )

            if epoch % config['samples']['samples_freq'] == 0 or epoch == config['train']['epochs'] - 1:
                # generator = torch.Generator(device=pipeline.device).seed() #.manual_seed(global_step)
                nsamp = 5 if 'samples_num' not in config['samples'] else config['samples']['samples_num']
                
                for g in range(nsamp):
                    labels = None if 'nclasses' not in config['dataset'] else torch.Tensor([g%config['dataset']['nclasses']]).to(device=pipeline.device)
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        batch_size=1, #config['dataset']['batch_size'],
                        num_inference_steps=config['diffuser']['num_inference_steps'],
                        output_type="numpy" # "pil"
                    ).images 
                    
                    # np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)
                    
                    if image_key == 'image' and 'segmentation' not in config['model']:
                        np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)

                        images = torch.from_numpy(images) 
                        latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = (images / 2 + 0.5).clamp(0, 1).cpu()
                        images = images.permute(0, 2, 3, 1)#.float().numpy()

                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                    elif image_key == 'segmentation' and 'segmentation' not in config['model']:
                        np.save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy", images)

                        images = torch.from_numpy(images) 
                        latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = images[:, 0, ...].unsqueeze(1) 
                        images = images.clamp(0, 1).cpu()
                        #images = images.permute(0, 2, 3, 1)#.float().numpy()
                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                    elif image_key == 'image' and config['model']['segmentation']:
                        n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
                        images_chann = torch.from_numpy(images[...,:n_channels])
                        masks_chann = torch.from_numpy(images[...,n_channels:])

                        np.save(f"{BASE_DIR}/samples/sample_{global_step:06d}-{g:02d}.npy", images_chann)
                        np.save(f"{BASE_DIR}/samples/mask_{epoch:04d}-{global_step:06d}-{g:02d}.npy", masks_chann)

                        latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                        images = AE.decode(latents, return_dict=False)[0] 
                        images = (images / 2 + 0.5).clamp(0, 1).cpu()
                        #images = images.permute(0, 2, 3, 1).float().numpy()
                        grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                        grid.save(f"{BASE_DIR}/samples/sample_{global_step:06d}_{g:02d}.jpg")

                        mask_latents = masks_chann.permute(0, 3, 1, 2).to('cuda')
                        mask = AE.decode(mask_latents, return_dict=False)[0]
                        mask = mask.mean(dim=1, keepdim=True)
                        #mask = mask[:, 0, ...].unsqueeze(1)
                        mask = mask.clamp(0, 1).cpu().numpy()
                        #mask = np.round(mask * 5).astype(int)
                        grid = T.ToPILImage(mode='RGB')(make_grid(torch.from_numpy(mask), nrow=1))
                        grid.save(f"{BASE_DIR}/samples/mask_{global_step:06d}_{g:02d}.jpg")

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

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(f"{BASE_DIR}/model")

                if 'ema' in config:
                    ema_model.restore(unet.parameters())


        last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break        

    logger.info(f"Finish training epoch = {epoch}")
    # Generate images with the last model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        pipeline = PIPELINES[config['diffuser']['pipeline_type']](
            unet=unet,
            scheduler=noise_scheduler,
        )      

        # generator = torch.Generator(device=pipeline.device).seed()
        for g in range(config['samples']['samples_gen']):
            labels = None if 'nclasses' not in config['dataset'] else torch.Tensor([g%config['dataset']['nclasses']]).to(device=pipeline.device)
     
            images = pipeline(
                generator=generator,
                batch_size=1, #config['dataset']['batch_size'],
                num_inference_steps=config['diffuser']['num_inference_steps'],
                output_type="numpy" #"pil"
            ).images

            if image_key == 'image' and 'segmentation' not in config['model']:
                np.save(f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy", images)

                images = torch.from_numpy(images) 
                latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()
                images = images.permute(0, 2, 3, 1)#.float().numpy()
                grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")
            
            elif image_key == 'segmentation' and 'segmentation' not in config['model']:
                np.save(f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy", images)

                images = torch.from_numpy(images) 
                latents = norm_inv(images.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = images.clamp(0, 1).cpu()
                #images = images.permute(0, 2, 3, 1)#.float().numpy()
                grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")

            elif image_key == 'image' and config['model']['segmentation']:
                n_channels =  config['model']['params']['out_channels'] - config['model']['segmentation']
                images_chann = torch.from_numpy(images[...,:n_channels])
                masks_chann = torch.from_numpy(images[...,n_channels:])

                np.save(f"{BASE_DIR}/final/sample_{global_step:06d}-{g:02d}.npy", images_chann)
                np.save(f"{BASE_DIR}/final/mask_{epoch:04d}-{global_step:06d}-{g:02d}.npy", masks_chann)

                latents = norm_inv(images_chann.permute(0, 3, 1, 2).to('cuda'))
                images = AE.decode(latents, return_dict=False)[0] 
                images = (images / 2 + 0.5).clamp(0, 1).cpu()
                #images = images.permute(0, 2, 3, 1).float().numpy()
                grid = T.ToPILImage(mode='RGB')(make_grid(images, nrow=1))
                grid.save(f"{BASE_DIR}/final/sample_{global_step:06d}_{g:02d}.jpg")

                mask = AE.decode(masks_chann.permute(0, 3, 1, 2).to('cuda'), return_dict=False)[0].clamp(0, 1).cpu().numpy()
                mask = np.round(mask * 5).astype(int)
                grid = T.ToPILImage(mode='RGB')(make_grid(torch.from_numpy(mask/5), nrow=1))
                grid.save(f"{BASE_DIR}/final/mask_{global_step:06d}_{g:02d}.jpg")

    accelerator.end_training()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)