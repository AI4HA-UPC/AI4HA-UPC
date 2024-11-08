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
from tqdm.auto import tqdm
import accelerate

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.models import TransformerTemporalModel
from packaging import version

from accelerate import DistributedDataParallelKwargs

from time import time
import numpy as np
from numpy.random import random

from ai4ha.util import save_config, time_management, \
    fix_paths, experiment_name_diffusion
from ai4ha.diffusion.pipelines.pipeline_ddpm_1d import DDPMPipeline
from ai4ha.diffusion.pipelines.pipeline_ddim import DDIMPipeline
from ai4ha.diffusion.models.unets.unet_1d import UNet1DModel
from ai4ha.diffusion.models.unets.cond_unet_1d import CondUNet1DModel
from ai4ha.diffusion.models.Transfusion import TransEncoder
from ai4ha.preprocess.augmentation import augmentation_mix_up
from ai4ha.log.textlog import textlog
from ai4ha.util.train import get_most_recent_checkpoint, \
    save_checkpoint_accelerate, get_diffuser_scheduler, get_optimizer
from ai4ha.util.sampling import sampling_diffusion
from ai4ha.util.misc import extract_into_tensor, log_cond
from ai4ha.util.data import load_dataset

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline
}

DIRS = ['checkpoints', 'logs', 'samples', "final", "model"]

logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    save_config(config, f"{BASE_DIR}/config.yaml")
    accparams = config['accelerator'].copy()
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(
            **config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    if 'class_embed_type' in config['model']['params']:
        class_conditioned = config['model']['params'][
            'class_embed_type'] is not None
    else:
        class_conditioned = False

    if 'num_class_embeds' in config['model']['params']:
        class_conditioned = config['model']['params'][
            'num_class_embeds'] is not None
    else:
        class_conditioned = False

    # Prepare the dataset
    # train_data = instantiate_from_config(config['dataset']['train'])
    # test_data = instantiate_from_config(config['dataset']['test'])
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                                **config["dataloader"])")

    dataloaders = load_dataset(config)
    len_train_data = len(dataloaders['train'])
    logger.info(f"Dataset size: {len_train_data}")

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
                if class_conditioned:
                    load_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "unet_ema"), CondUNet1DModel)
                else:
                    load_model = EMAModel.from_pretrained(
                        os.path.join(input_dir, "unet_ema"), UNet1DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if class_conditioned:
                    if config['model']['modeltype'] == "UNET1C":
                        load_model = CondUNet1DModel.from_pretrained(
                            input_dir, subfolder="unet")
                    elif config['model']['modeltype'] == "Transfusion":
                        load_model = TransEncoder.from_pretrained(
                            input_dir, subfolder="unet")
                else:
                    if config['model']['modeltype'] == "UNET1":
                        load_model = UNet1DModel.from_pretrained(
                            input_dir, subfolder="unet")
                    elif config['model']['modeltype'] == "Transfusion":
                        load_model = TransEncoder.from_pretrained(
                            input_dir, subfolder="unet")

                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    global_step = 0
    first_epoch = 0
    # Get the most recent checkpoint
    path = get_most_recent_checkpoint(logger, BASE_DIR)
    logger.info(f'CHECKPOINT: {path}')

    total_batch_size = config['dataloader']['batch_size'] * \
        accelerator.num_processes * accparams['gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(
        len_train_data / accparams['gradient_accumulation_steps'])
    max_train_steps = config['train']['num_epochs'] * num_update_steps_per_epoch

    resume_from_checkpoint = True
    if path is None:
        logger.info("Checkpoint does not exist. Starting a new training run.")
        resume_from_checkpoint = None
        global_step = first_epoch = 0
    else:
        global_step = int(
            path.split("_")[1])  # *  config['train']['checkpoint_freq']
        # resume_global_step = global_step  #* accparams['gradient_accumulation_steps']
        first_epoch = global_step  # // num_update_steps_per_epoch
        # * accparams['gradient_accumulation_steps']))
        # resume_step = (resume_global_step % (num_update_steps_per_epoch))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    tlog = textlog(f"{BASE_DIR}/logs/losslog.csv", [f"loss ({config['loss']['loss']})"])

    if config['model']['modeltype'] == "TTRANSFORMER":
        model = TransformerTemporalModel(**config['model']['params'])
    elif config['model']['modeltype'] == "UNET1C":
        model = CondUNet1DModel(**config['model']['params'])
    elif config['model']['modeltype'] == "UNET1":
        model = UNet1DModel(**config['model']['params'])
    elif config['model']['modeltype'] == "Transfusion":
        model = TransEncoder(**config['model']['params'])
    else:
        raise ValueError(
            f"Unsupported model type: {config['model']['modeltype']}")

    if class_conditioned:
        log_cond(logger, 'class_embed_type', config['model']['params'])
        log_cond(logger, 'num_class_embeds', config['model']['params'])

    if class_conditioned and "nclasses" not in config['dataset']:
        raise ValueError(
            "Class conditioning is enabled but the number of classes is not specified"
        )
    # model.disable_gradient_checkpointing()

    # Create EMA for the model.
    if 'ema' in config:
        ema_model = EMAModel(
            model.parameters(),
            decay=config['ema']['max_decay'],
            use_ema_warmup=True,
            inv_gamma=config['ema']['inv_gamma'],
            power=config['ema']['power'],
            model_cls=CondUNet1DModel if class_conditioned else UNet1DModel,
            model_config=model.config,
        )

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    # Initialize the optimizer
    optimizer = get_optimizer(model, accelerator, config)

    lr_scheduler = get_scheduler(
        config['lr_scheduler']['type'],
        optimizer=optimizer,
        num_warmup_steps=config['lr_scheduler']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len_train_data * config['train']['num_epochs']))

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloaders['train'], lr_scheduler)

    if 'ema' in config:
        ema_model.to(accelerator.device)

    logger.info("***** Running training *****")
    logger.info(
        f"  Num examples = {len_train_data * config['dataloader']['batch_size']}"
    )
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

    if resume_from_checkpoint:
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        logger.info(
            f"Resuming from checkpoint {path} - Resume epoch: {global_step}")

    # hps = {
    #     "num_iterations":
    #     config['train']['num_epochs'],
    #     "learning_rate":
    #     config['optimizer']['learning_rate'] * accelerator.num_processes
    # }
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    last_time = time()
    qe_time = []

    time_budget = 2000
    margin = 1.5
    if 'time' in config:
        if config['local']:
            time_budget = config['time'] * 60
        else:
            if 'sample_time' in config['samples']:
                margin = config['samples']['sample_time']
            else:
                margin = 1.5
            # 90 minutes of margin in the cluster for generating samples
            time_budget = (config['time'] * 60) - (margin * 60)

    logger.info(
        f"Training TIME: {time_budget/60} hours| margin: {margin} hours")

    log_cond(logger, 'time_embedding_type', config['model']['params'])
    log_cond(logger, 'class_embed_type', config['model']['params'])

    if 'augmentation' in config['train']:
        mixup = 'mixup' in config['train']['augmentation']
        if mixup:
            coef, freq = config['train']['augmentation']['mixup'][
                'coef'], config['train']['augmentation']['mixup']['freq']
            logger.info(f"Mixup: {mixup} - Coef: {coef} - Freq: {freq}")
    else:
        mixup = False

    # Train!
    if first_epoch > config['train']['num_epochs']:
        epoch = config['train']['num_epochs']
    generator = torch.Generator(device=accelerator.device)

    # Select the loss function
    if config['loss']['loss'] == "L2":
        loss_fn = F.mse_loss
    elif config['loss']['loss'] == "L1":
        loss_fn = F.l1_loss
    else:
        raise ValueError(f"Unsupported loss function: {config['loss']['loss']}")

    for epoch in range(first_epoch, config['train']['num_epochs']):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % accparams['gradient_accumulation_steps'] == 0:
            #         progress_bar.update(1)
            #     continue

            clean_images, labels = batch
            clean_images = clean_images.to(dtype=torch.float32)
            if mixup and random() < freq:
                clean_images = augmentation_mix_up(clean_images, labels, coef)

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps, (bsz, ),
                device=clean_images.device)

            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if class_conditioned:
                    labels = labels.to(dtype=torch.long).to(
                        clean_images.device)
                    model_output = model(noisy_images,
                                         timesteps,
                                         class_labels=labels).sample
                else:
                    model_output = model(noisy_images, timesteps).sample

                if config['diffuser']['prediction_type'] == "epsilon":
                    # this could have different weights!
                    loss = loss_fn(model_output, noise)
                elif config['diffuser']['prediction_type'] == "sample":
                    alpha_t = extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps,
                        (clean_images.shape[0], 1, 1, 1))
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * loss_fn(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
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

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if 'ema' in config:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                # if (epoch + 1) % config['train']['checkpoint_freq'] == 0:
                #     if accelerator.is_main_process:
                #         save_checkpoint_accelerate(logger, BASE_DIR, config,
                #                                    accelerator, epoch)

            logs = {
                "loss": loss.detach().item(),
                "m_loss": mean_loss / (step + 1),
                "step": global_step
            }

            if 'ema' in config:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)

        lr_scheduler.step()
        logger.info(
            f"Epoch {epoch} - Mean Loss: {mean_loss / (step + 1)} - LR: {optimizer.param_groups[0]['lr']}"
        )
        tlog.write({f"loss ({config['loss']['loss']})": mean_loss / (step + 1)})
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate samples for visual inspection
        if accelerator.is_main_process:
            # accelerator.get_tracker("wandb").log(
            #     {
            #         "loss": loss.detach().item(),
            #         "lr": lr_scheduler.get_last_lr()[0]
            #     },
            #     step=global_step)

            # save the model
            if ((epoch + 1) % config['train']['checkpoint_epoch_freq']
                    == 0) or epoch == (config['train']['num_epochs'] - 1):
                save_checkpoint_accelerate(logger, BASE_DIR, config,
                                           accelerator, epoch)

            if (epoch + 1
                ) % config['samples']['samples_freq'] == 0 or epoch == config[
                    'train']['num_epochs'] - 1:
                unet = accelerator.unwrap_model(model)
                if 'ema' in config:
                    ema_model.copy_to(unet.parameters())

                pipeline = PIPELINES[config['diffuser']['type']](
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(f"{BASE_DIR}/model")

                nsamp = 5 if 'samples_num' not in config[
                    'samples'] else config['samples']['samples_num']
                logger.info(f"Checkpoint sampling - epoch {epoch}")
                sampling_diffusion(config, BASE_DIR, pipeline, generator,
                                   class_conditioned, nsamp, epoch, False)

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break

    save_checkpoint_accelerate(logger, BASE_DIR, config, accelerator,
                               epoch + 1)
    tlog.close()
    logger.info(f"Finish training epoch = {epoch}")
    # Generate images with the last model and save it
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        pipeline = PIPELINES[config['diffuser']['type']](
            unet=unet,
            scheduler=noise_scheduler,
        )

        pipeline.save_pretrained(f"{BASE_DIR}/model")

        nsamp = config['samples']['samples_gen'] // config['dataloader'][
            'batch_size']
        logger.info("Final sampling")
        sampling_diffusion(config, BASE_DIR, pipeline, generator,
                           class_conditioned, nsamp, epoch, True)

    logger.info("Finishing training")
    accelerator.end_training()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def timeDiffusionTrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])

    cfg['name'] = experiment_name_diffusion(cfg)
    print(cfg['name'])
    main(cfg)


if __name__ == "__main__":
    timeDiffusionTrain()
