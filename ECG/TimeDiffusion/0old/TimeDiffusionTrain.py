import argparse
import inspect
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
import accelerate

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import DistributedDataParallelKwargs

from time import time
import numpy as np
from numpy.random import random
import shutil
from ai4ha.util import instantiate_from_config, load_config_fix_paths, \
    time_management
from ai4ha.diffusion.pipelines.pipeline_ddpm_1d import DDPMPipeline
from ai4ha.diffusion.pipelines.pipeline_ddim import DDIMPipeline
from ai4ha.diffusion.models.unet_1d import UNet1DModel
from ai4ha.diffusion.models.cond_unet_1d import CondUNet1DModel
from ai4ha.preprocess.augmentation import augmentation_mix_up
from ai4ha.log.textlog import textlog
from diffusers.models import TransformerTemporalModel
from packaging import version

SCHEDULERS = {
    'DDPM': DDPMScheduler,
    'DDIM': DDIMScheduler
}

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline
}

DIRS = ['checkpoints', 'logs', 'samples', "final", "model"]


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.14.0.dev0")

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
    res = arr[timesteps.cpu()].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_diffuser_scheduler(config):
    scheduler = SCHEDULERS[config['diffuser']['type']]

    if 'clip_sample' not in config['diffuser']:
        clip = True
    else:
        clip = config['diffuser']['clip_sample']
    params = {
        'num_train_timesteps': config['diffuser']['num_steps'],
        'beta_schedule': config['diffuser']['beta_schedule'],
        'clip_sample': clip
    }

    if "prediction_type" in set(inspect.signature(scheduler.__init__).parameters.keys()):
        params['prediction_type'] = config['diffuser']['prediction_type']
    if ("variance_type" in set(inspect.signature(scheduler.__init__).parameters.keys())) and ("variance_type" in config['diffuser']):
        params['variance_type'] = config['diffuser']['variance_type']
    if "betas" in config['diffuser']:
        params['beta_start'] = config['diffuser']['betas'][0]
        params['beta_end'] = config['diffuser']['betas'][1]

    return scheduler(**params)


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    accparams = config['accelerator']
    # accparams["logging_dir"] = f"{BASE_DIR}/logs"
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(
            **config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    class_conditioned = config['model']['params'][
        'class_embed_type'] is not None

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
                    load_model = CondUNet1DModel.from_pretrained(
                        input_dir, subfolder="unet")
                else:
                    load_model = UNet1DModel.from_pretrained(input_dir,
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

    class_conditioned = config['model']['params'][
        'class_embed_type'] is not None

    if config['model']['modeltype'] == "TTRANSFORMER":
        model = TransformerTemporalModel(**config['model']['params'])
    elif config['model']['modeltype'] == "UNET1":
        if class_conditioned:
            model = CondUNet1DModel(**config['model']['params'])
        else:
            model = UNet1DModel(**config['model']['params'])

    logger.info(
        f"CLASS CONDITIONING: {class_conditioned}, {config['model']['params']['class_embed_type']}"
    )
    if class_conditioned and "nclasses" not in config['dataset']:
        raise ValueError(
            "Class conditioning is enabled but the number of classes is not specified"
        )
    model.disable_gradient_checkpointing()

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr_scheduler']['learning_rate'] * accelerator.num_processes,
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        weight_decay=config['optimizer']['weight_decay'],
        eps=config['optimizer']['epsilon'],
    )

    # Prepare the dataset
    train_data = instantiate_from_config(config['dataset']['train'])
    # test_data = instantiate_from_config(config['dataset']['test'])

    if 'sampling_smoothing' in config['dataset']['train']['params']:
        sampler = WeightedRandomSampler(
            train_data.weights, len(train_data), replacement=True)
        config['dataset']["dataloader"]["sampler"] = sampler
        config['dataset']["dataloader"]["shuffle"] = False
        logger.info(f"Using weighted sampler")
        logger.info(f"Weights: {train_data.weights}")
    else:
        config['dataset']["dataloader"]["shuffle"] = True
        logger.info(f"Using normal sampler")

    logger.info(f"dataloader: {config['dataset']['dataloader']}")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, **config['dataset']["dataloader"])

    logger.info(f"Dataset size: {len(train_data)}")

    lr_scheduler = get_scheduler(
        config['lr_scheduler']['type'],
        optimizer=optimizer,
        num_warmup_steps=config['lr_scheduler']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) * config['train']['epochs']),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    if 'ema' in config:
        ema_model.to(accelerator.device)

    total_batch_size = config['dataset']['dataloader']['batch_size'] * \
        accelerator.num_processes * accparams['gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accparams['gradient_accumulation_steps'])
    max_train_steps = config['train']['epochs'] * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num Epochs = {config['train']['epochs']}")
    logger.info(
        f"  Instantaneous batch size per device = {config['dataset']['dataloader']['batch_size']}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}"
    )
    if 'sampling_smoothing' in config['dataset']['train']['params']:
        logger.info(f"  Using weighted sampler")
        logger.info(f"  Weights: {train_data.weights}")
        print(f"  Weights: {train_data.weights}")
    else:
        logger.info(f"  Using normal sampler")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Get the most recent checkpoint
    print(f'{BASE_DIR}/checkpoints/')
    dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    # print(dirs)
    if dirs != []:
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = None

    logger.info(f'CHECKPOINT: {path}')

    resume_from_checkpoint = True
    if path is None:
        accelerator.print(
            f"Checkpoint does not exist. Starting a new training run.")
        resume_from_checkpoint = None
        resume_step = 0
    else:
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        global_step = int(path.split("_")[1]) * \
            config['train']['checkpoint_freq']
        resume_global_step = global_step * \
            accparams['gradient_accumulation_steps']
        first_epoch = global_step // num_update_steps_per_epoch
        # * accparams['gradient_accumulation_steps']))
        resume_step = (resume_global_step % (num_update_steps_per_epoch))
        accelerator.print(
            f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}"
        )

    hps = {
        "num_iterations":
        config['train']['epochs'],
        "learning_rate":
        config['lr_scheduler']['learning_rate'] * accelerator.num_processes
    }
    accelerator.init_trackers(
        config['name'],
        config=hps,
        init_kwargs={"wandb": {
            "dir": os.path.join(BASE_DIR, "logs")
        }})
    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = (config['time'] * 60) - 90  # 90 minutes of margin
    else:
        time_budget = 2000

    # Train!
    logger.info(
        f"time_embedding_type: {config['model']['params']['time_embedding_type']}"
    )
    logger.info(
        f"class_embed_type: {config['model']['params']['class_embed_type']}")
    if 'augmentation' in config['train']:
        mixup = 'mixup' in config['train']['augmentation']
        if mixup:
            coef, freq = config['train']['augmentation']['mixup'][
                'coef'], config['train']['augmentation']['mixup']['freq']
            logger.info(f"Mixup: {mixup} - Coef: {coef} - Freq: {freq}")
    else:
        mixup = False
    generator = torch.Generator(device=accelerator.device)
    for epoch in range(first_epoch, config['train']['epochs']):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            # logger.info(f"*{epoch} - {step}")
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % accparams['gradient_accumulation_steps'] == 0:
                    progress_bar.update(1)
                continue

            clean_images, labels = batch
            clean_images = clean_images.to(dtype=torch.float32)

            if mixup and random() < freq:
                clean_images = augmentation_mix_up(clean_images, labels, coef)
            # logger.info(clean_images.shape, labels.shape)
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps, (bsz, ),
                device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if class_conditioned:
                    labels = labels.to(dtype=torch.float32).to(
                        clean_images.device)
                    model_output = model(noisy_images,
                                         timesteps,
                                         class_labels=labels).sample
                else:
                    model_output = model(noisy_images, timesteps).sample

                if config['diffuser']['prediction_type'] == "epsilon":
                    # this could have different weights!
                    loss = F.mse_loss(model_output, noise)
                elif config['diffuser']['prediction_type'] == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps,
                        (clean_images.shape[0], 1, 1, 1))
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
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
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if 'ema' in config:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if (global_step + 1) % config['train']['checkpoint_freq'] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            f'{BASE_DIR}/checkpoints/',
                            f"checkpoint_{global_step//config['train']['checkpoint_freq']:06d}"
                        )
                        # save_path = f'{BASE_DIR}/checkpoints/'
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
                        dirs = sorted(
                            [d for d in dirs if d.startswith("checkpoint")])
                        if len(dirs) > config["projectconf"]["total_limit"]:
                            for d in dirs[:-config["projectconf"]
                                          ["total_limit"]]:
                                logger.info(
                                    f'delete {BASE_DIR}/checkpoints/{d}')
                                shutil.rmtree(f'{BASE_DIR}/checkpoints/{d}')

            logs = {
                "loss": loss.detach().item(),
                "m_loss": mean_loss / (step + 1),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }

            if 'ema' in config:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)

        tlog.write({"MSEloss": mean_loss / (step + 1)})
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate samples for visual inspection
        if accelerator.is_main_process:
            accelerator.get_tracker("wandb").log(
                {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                },
                step=global_step)
            unet = accelerator.unwrap_model(model)
            if 'ema' in config:
                ema_model.copy_to(unet.parameters())

            pipeline = PIPELINES[config['diffuser']['type']](
                unet=unet,
                scheduler=noise_scheduler,
            )

            if (epoch + 1
                ) % config['samples']['samples_freq'] == 0 or epoch == config[
                    'train']['epochs'] - 1:
                nsamp = 5 if 'samples_num' not in config[
                    'samples'] else config['samples']['samples_num']

                samples, samples_labs = [], []

                for g in range(nsamp * config['dataset']['nclasses']):
                    labels = None if not class_conditioned else torch.Tensor(
                        [g % config['dataset']['nclasses']]).to(
                            device=pipeline.device)
                    # run pipeline in inference (sample random noise and denoise)
                    data = pipeline(
                        generator=generator,
                        class_cond=labels,
                        batch_size=1,
                        num_inference_steps=config['diffuser']
                        ['num_inference_steps'],
                        output_type="numpy"  # "pil"
                    ).images

                    # logger.info(f'LABEL: {labels}')
                    samples.append(data.cpu().numpy())
                    if class_conditioned:
                        samples_labs.append(labels.cpu().numpy())
                    else:
                        samples_labs.append(0)
                    # images[0].save(f"{BASE_DIR}/samples/samples_{epoch:04d}-{g:02d}.jpg")
                np.savez(f"{BASE_DIR}/samples/sampled_data_{epoch:05d}.npz",
                         samples=np.array(samples),
                         classes=np.array(samples_labs))

            if (epoch % config['train']['checkpoint_epoch_freq']
                    == 0) or epoch == (config['train']['epochs'] - 1):
                # save the model
                pipeline.save_pretrained(f"{BASE_DIR}/model")

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break

    tlog.close()
    logger.info(f"Finish training epoch = {epoch}")
    # Generate images with the last model
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(model)
        pipeline = PIPELINES[config['diffuser']['type']](
            unet=unet,
            scheduler=noise_scheduler,
        )

        samples, samples_labs = [], []
        for g in range(config['samples']['samples_gen'] //
                       config['dataset']['dataloader']['batch_size']):
            labels = None if not class_conditioned else torch.randint(
                0, config['dataset']['nclasses'],
                (config['dataset']['dataloader']['batch_size'], )).to(
                    device=pipeline.device)
            data = pipeline(
                generator=generator,
                class_cond=labels,
                batch_size=config['dataset']['dataloader']['batch_size'],
                num_inference_steps=config['diffuser']['num_inference_steps'],
                output_type="numpy",
            ).images

            samples.append(data.cpu().numpy())
            samples_labs.append(labels.cpu().numpy())
        np.save(f"{BASE_DIR}/samples/samples.npy", np.concatenate(samples))
        np.save(f"{BASE_DIR}/samples/labels.npy", np.concatenate(samples_labs))

    accelerator.end_training()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='configuration file')
    parser.add_argument('--local',
                        action='store_true',
                        default=False,
                        help='Runned locally')
    parser.add_argument("--hours",
                        default=None,
                        type=int,
                        help="override time limit for jobs")

    args = parser.parse_args()
    config = load_config_fix_paths(args.config, args.local)
    if args.hours is not None:
        config["time"] = args.hours
    main(config)
