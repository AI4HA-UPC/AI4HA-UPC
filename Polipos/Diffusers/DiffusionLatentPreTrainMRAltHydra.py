################################################################################
# Script for training a diffusion model with latent space using the Diffusers  #
# library and Hydra.                                                           #
#                                                                              #
# The dataset has to be already preprocessed using a VAE and stored in a folder#
# The augmentation data has to be included in the dataset if needed.           #
#                                                                              #
#  MULTI RESOLUTION VERSION - TRAINS WITH DATA AT DIFFERENT RESOLUTIONS        #
################################################################################
import logging
import math
import os
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import accelerate
from tqdm.auto import tqdm
import torchvision.transforms as T

from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs

from time import time
import numpy as np

from ai4ha.diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from ai4ha.diffusion.pipelines.pipeline_ddim import DDIMPipeline

from ai4ha.log.textlog import textlog
from ai4ha.Autoencoders import AutoencoderKL
from ai4ha.util import instantiate_from_config, \
        time_management, fix_paths, experiment_name_diffusion, \
        save_config  # , print_memory
from ai4ha.util.train import get_most_recent_checkpoint, \
    save_checkpoint_accelerate, get_optimizer, get_lr_scheduler
from ai4ha.util.train import get_diffuser_scheduler
from ai4ha.util import dataset_statistics, dataset_first_batch_std
from ai4ha.diffusion.losses.diffusion import loss_weighting
from ai4ha.util.misc import extract_into_tensor
from packaging import version

SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler
}

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline
}

LATENTS = {
    "AEKL": AutoencoderKL
}

DIRS = ['checkpoints', 'logs', 'samples', "final", "model"]


logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    # Check if we are using a bootstrap model for intializing the weights
    if 'bootstrap_model' in config['dataset']:
        BOOT_DIR = f"{config['exp_dir']}/logs/{config['dataset']['bootstrap_model']}"
    else:
        BOOT_DIR = None

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    save_config(config, f"{BASE_DIR}/config.yaml")
    accparams = config['accelerator']
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(
            **config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if 'ema' in config:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if 'ema' in config:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir,
                                                         subfolder="unet")
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
    tlog = textlog(f"{BASE_DIR}/logs/losslog.csv", ["MSEloss"])

    # DATA LOADERs
    train_data1 = instantiate_from_config(config['dataset']['train1'])
    sample_size1 = config['dataset']['train1']['sample_size']
    train_dataloader1 = torch.utils.data.DataLoader(train_data1,
                                                    **config["dataloader"])
    train_data2 = instantiate_from_config(config['dataset']['train2'])
    sample_size2 = config['dataset']['train2']['sample_size']

    train_dataloader2 = torch.utils.data.DataLoader(train_data2,
                                                    **config["dataloader"])
    # if 'test' in config['dataset']:
    #     test_data = instantiate_from_config(config['dataset']['test'])
    # if 'test' in config['dataset']:
    #     test_dataloader = torch.utils.data.DataLoader(test_data,
    #                                                   **config["dataloader"])
    logger.info(f"Dataset size: {len(train_data1)} ---- {len(train_data2)}")

    # Get the most recent checkpoint path if it exists
    path = get_most_recent_checkpoint(logger, BASE_DIR)

    model = UNet2DModel(**config['model']['params'])

    # If there is no checkpoint, we can load a bootstrap model
    if path is None and BOOT_DIR is not None:
        boot_path = get_most_recent_checkpoint(logger, BOOT_DIR)
        if boot_path is not None:
            logger.info(f"Loading bootstrap model from {boot_path}")
            model = UNet2DModel.from_pretrained(os.path.join(
                f'{BOOT_DIR}/checkpoints/', boot_path),
                                                subfolder="unet")
            model.config.sample_size = config['model']['params']['sample_size']
            first_epoch = 0

    # Create EMA for the model.
    if 'ema' in config:
        ema_model = EMAModel(
            model.parameters(),
            decay=config['ema']['max_decay'],
            use_ema_warmup=True,
            inv_gamma=config['ema']['inv_gamma'],
            power=config['ema']['power'],
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    # Initialize the optimizer
    optimizer = get_optimizer(model, accelerator, config)

    logger.info(f"* LR scheduler= {config['lr_scheduler']['type']}")
    lr_scheduler = get_lr_scheduler(config, optimizer, train_dataloader1,
                                    accparams)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader1, train_dataloader2, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader1, train_dataloader2, lr_scheduler)

    if 'ema' in config:
        ema_model.to(accelerator.device)

    total_batch_size = config['dataloader'][
        'batch_size'] * accelerator.num_processes * accparams[
            'gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(
        (len(train_dataloader1) + len(train_dataloader2)) /
        accparams['gradient_accumulation_steps'])
    max_train_steps = config['train']['num_epochs'] * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data1) + len(train_data2)}")
    logger.info(f"  Num Epochs = {config['train']['num_epochs']}")
    logger.info(
        f"  Instantaneous batch size per device = {config['dataloader']['batch_size']}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    resume_from_checkpoint = True
    if path is None:
        accelerator.print(
            "Checkpoint does not exist. Starting a new training run.")
        resume_from_checkpoint = None
        resume_step = 0
        first_epoch = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        global_step = int(path.split("_")[1])

        # resume_global_step = global_step
        first_epoch = global_step
        accelerator.print(
            f"Resuming from checkpoint {path} - Resume step: {global_step}")

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = config['time'] * 60
    else:
        time_budget = 2000

    # Train!
    image_key = 'image' if "image_key" not in config['model'] else config[
        'model']['image_key']
    generator = torch.Generator(device=accelerator.device)

    # Compute dataset statistics when scaling
    if 'std_scaling' in config['train']:
        if 'segmentation' in config['model']:
            in_channels = config['model']['params']["in_channels"] - config[
                'model']['segmentation']
        else:
            in_channels = config['model']['params']["in_channels"]
        if config['train']['std_scaling'] == "full":
            xmean, xstd = dataset_statistics(train_dataloader1,
                                             image_key,
                                             channels=in_channels)
            norm_tfm = T.Normalize(xmean, xstd)
        elif config['train']['std_scaling'] == "batch":
            xstd = dataset_first_batch_std(train_data1,
                                           config['dataset']["dataloader"],
                                           image_key)
            norm_tfm = T.Normalize(1.0, xstd)
        elif 'std_scaling_val' in config['train']:
            norm_tfm = T.Normalize(1.0, config['train']['std_scaling_val'])
        else:
            norm_tfm = T.Normalize(1.0, 0.18215)
    else:
        norm_tfm = T.Normalize(1.0, 0.18215)

    model.train()
    num_update_steps_per_epoch = math.ceil(
                    (len(train_dataloader1) + len(train_dataloader2)) /
                    accparams['gradient_accumulation_steps'])
    for epoch in range(first_epoch, config['train']['num_epochs']):
        mean_loss = 0
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (batch1, batch2) in enumerate(
                zip(train_dataloader1, train_dataloader2)):
            for nb in range(2):
                if nb == 0:
                    batch = batch1
                    model.config.sample_size = sample_size1
                else:
                    batch = batch2
                    model.config.sample_size = sample_size2

                if 'std_scaling' in config['train'] and config['train'][
                        'std_scaling']:
                    clean_images_latents = norm_tfm(batch[image_key])
                else:
                    clean_images_latents = batch[image_key]

                if "segmentation" in config['model']:
                    if config['model']['segmentation'] > 0:
                        clean_images_latents = torch.cat(
                            (clean_images_latents, batch['segmentation']), 1)

                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images_latents.shape).to(
                    clean_images_latents.device)
                bsz = clean_images_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=clean_images_latents.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(
                    clean_images_latents, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    # labels = None if 'class_ohe' not in batch else torch.squeeze(batch['class_ohe'])
                    labels = None if 'class_label' not in batch else batch[
                        'class_label']

                    model_output = model(noisy_images,
                                         timesteps,
                                         class_labels=labels).sample

                    if 'weighting' in config['diffuser']:
                        alpha_t = extract_into_tensor(
                            noise_scheduler.alphas_cumprod.to(
                                clean_images_latents.device), timesteps,
                            (clean_images_latents.shape[0], 1, 1, 1))
                        snr_weights = alpha_t / (1 - alpha_t)
                        e_weights = loss_weighting(
                            config['diffuser']['prediction_type'],
                            config['diffuser']['weighting']['type'],
                            snr_weights, timesteps)
                    else:
                        e_weights = torch.ones_like(timesteps)

                    if config['diffuser']['prediction_type'] == "epsilon":
                        # loss = e_weights * F.mse_loss(
                        #     model_output,
                        #     noise)  # this could have different weights!
                        loss = F.mse_loss(
                            model_output,
                            noise)  # this could have different weights!
                    elif config['diffuser']['prediction_type'] == "sample":
                        alpha_t = extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps,
                            (clean_images_latents.shape[0], 1, 1, 1))
                        e_weights = alpha_t / (1 - alpha_t)
                        loss = e_weights * F.mse_loss(model_output,
                                                      clean_images_latents,
                                                      reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(
                            f"Unsupported prediction type: {config['diffuser']['prediction_type']}"
                        )
                    mean_loss += loss.detach().item()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)

        lr_scheduler.step()
        progress_bar.close()

        if accelerator.is_main_process:
            if ((epoch + 1) % config['train']['checkpoint_epoch_freq']
                    == 0) or (epoch == (config['train']['num_epochs'] - 1)):
                save_checkpoint_accelerate(logger, BASE_DIR, config,
                                           accelerator, epoch)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if 'ema' in config:
                ema_model.step(model.parameters())

        if 'ema' in config:
            logs["ema_decay"] = ema_model.cur_decay_value

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            progress_bar.close()
            tlog.write({"MSEloss": mean_loss / (step + 1)})

            # accelerator.get_tracker("wandb").log(
            #     {
            #         "loss": loss.detach().item(),
            #         "lr": lr_scheduler.get_last_lr()[0]
            #     },
            #     step=global_step)
            unet = accelerator.unwrap_model(model)
            if 'ema' in config:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            pipeline = PIPELINES[config['diffuser']['type']](
                unet=unet,
                scheduler=noise_scheduler,
            )

            # Generate some samples
            if epoch % config['samples'][
                    'samples_freq'] == 0 or epoch == config['train'][
                        'num_epochs'] - 1:

                nsamp = 5 if 'samples_num' not in config[
                    'samples'] else config['samples']['samples_num']

                for g in range(nsamp):
                    labels = None if 'nclasses' not in config[
                        'dataset'] else torch.Tensor(
                            [g % config['dataset']['nclasses']]).to(
                                device=pipeline.device)
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        class_cond=labels,
                        batch_size=1,  # config['diffuser']['batch_size'],
                        num_inference_steps=config['diffuser']
                        ['num_inference_steps'],
                        output_type="numpy",  # "pil"
                        latent=True,
                        vae=None).images

                    if image_key == 'image' and 'segmentation' not in config[
                            'model']:
                        np.save(
                            f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy",
                            images)
                    elif image_key == 'image' and config['model'][
                            'segmentation']:
                        n_channels = config['model']['params'][
                            'out_channels'] - config['model']['segmentation']
                        images_chann = torch.from_numpy(
                            images[..., :n_channels])
                        np.save(
                            f"{BASE_DIR}/samples/samples_{epoch:04d}-{global_step:06d}-{g:02d}.npy",
                            images_chann)
                        masks_chann = torch.from_numpy(images[...,
                                                              n_channels:])
                        np.save(
                            f"{BASE_DIR}/samples/masks_{epoch:04d}-{global_step:06d}-{g:02d}.npy",
                            masks_chann)

            if 'ema' in config:
                ema_model.restore(unet.parameters())

            # save some model checkpoints during training
            if (((epoch % config['train']['checkpoint_freq']) + 1)
                    == 0) or epoch == (config['train']['num_epochs'] - 1):

                unet = accelerator.unwrap_model(model)
                if 'ema' in config:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(f"{BASE_DIR}/model{epoch:04d}")

                if 'ema' in config:
                    ema_model.restore(unet.parameters())

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break

    logger.info(f"Finish training epoch = {epoch}")

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        save_checkpoint_accelerate(logger, BASE_DIR, config, accelerator,
                                   epoch + 1)

    # Generate images with the last model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        pipeline = PIPELINES[config['diffuser']['type']](
            unet=unet,
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(f"{BASE_DIR}/model")

        # generator = torch.Generator(device=pipeline.device).seed()
        for g in range(config['samples']['samples_gen']):
            labels = None if 'nclasses' not in config[
                'dataset'] else torch.Tensor(
                    [g %
                     config['dataset']['nclasses']]).to(device=pipeline.device)

            images = pipeline(
                generator=generator,
                class_cond=labels,
                batch_size=1,  # config['dataset']['batch_size'],
                num_inference_steps=config['diffuser']['num_inference_steps'],
                output_type="numpy",  # "pil"
                latent=True,
                vae=None).images

            if image_key == 'image' and 'segmentation' not in config['model']:
                np.save(
                    f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy",
                    images)
            elif image_key == 'image' and config['model']['segmentation']:
                n_channels = config['model']['params'][
                    'out_channels'] - config['model']['segmentation']
                images_chann = torch.from_numpy(images[..., :n_channels])
                np.save(
                    f"{BASE_DIR}/final/samples_{global_step:06d}-{g:02d}.npy",
                    images_chann)
                masks_chann = torch.from_numpy(images[..., n_channels:])
                np.save(
                    f"{BASE_DIR}/final/masks_{global_step:06d}-{g:02d}.npy",
                    masks_chann)

    accelerator.end_training()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def LatentDiffusionTrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    cfg['name'] = experiment_name_diffusion(cfg)
    main(cfg)


if __name__ == "__main__":
    LatentDiffusionTrain()
