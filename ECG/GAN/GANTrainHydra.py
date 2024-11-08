import os
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode

import torch
# import torch.optim as optim
import logging
from time import time

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from accelerate import DistributedDataParallelKwargs

from ai4ha.util import instantiate_from_config, fix_paths, \
    time_management, experiment_name_GAN, save_config
from ai4ha.GAN.losses import calc_gradient_penalty
from ai4ha.log.textlog import textlog
from ai4ha.util.train import get_most_recent_checkpoint, get_best_loss, \
    save_best_model, save_checkpoint_accelerate, get_optimizer
from ai4ha.util.sampling import sampling_GAN


DIRS = ["checkpoints", "logs", "samples", "final", "model", "best"]
logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    save_config(config, f"{BASE_DIR}/config.yaml")

    accparams = config["accelerator"]
    # accparams["logging_dir"] = f"{BASE_DIR}/logs"
    accparams["project_dir"] = BASE_DIR

    if "projectconf" in config:
        accparams["project_config"] = ProjectConfiguration(
            **config["projectconf"])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams["gradient_accumulation_steps"] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    tlog = textlog(f"{BASE_DIR}/logs/losslog.csv",
                   ["lossD", "lossG", "loss_real", "loss_fake", "gp"])

    # Load the dataset
    train_data = instantiate_from_config(config['dataset']['train'])
    # test_data = instantiate_from_config(config['dataset']['test'])
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   **config["dataloader"])
    # Get the most recent checkpoint
    path = get_most_recent_checkpoint(logger, BASE_DIR)

    if path is not None:
        global_epoch = int(path.split("_")[1]) + 1
        resume_epoch = global_epoch
    else:
        global_epoch = 0
        resume_epoch = 0

    # Instantiate the model from the config file classes and parameters
    netG = instantiate_from_config(config['generator'])
    netD = instantiate_from_config(config['discriminator'])

    # Optimizers
    optimizerG = get_optimizer(netG, accelerator, config)
    optimizerD = get_optimizer(netD, accelerator, config)

    # Learning rate schedulers
    lr_schedulerD = get_scheduler(
        config['lr_scheduler']['type'],
        optimizer=optimizerD,
        num_warmup_steps=config['lr_scheduler']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) *
                            config['train']['num_epochs']))
    lr_schedulerG = get_scheduler(
        config['lr_scheduler']['type'],
        optimizer=optimizerG,
        num_warmup_steps=config['lr_scheduler']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) *
                            config['train']['num_epochs']))

    # Accelerator magic
    netD, netG, optimizerD, optimizerG, train_dataloader, lr_schedulerD, \
        lr_schedulerG = accelerator.prepare(netD, netG, optimizerD,
                                            optimizerG, train_dataloader,
                                            lr_schedulerD, lr_schedulerG)

    # Up to the GPU
    netD.to(accelerator.device)
    netG.to(accelerator.device)

    num_update_steps_per_epoch = len(train_dataloader) // config[
        'accelerator']['gradient_accumulation_steps']

    if global_epoch == 0:
        accelerator.print(
            "Checkpoint does not exist. Starting a new training run.")
    else:
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        accelerator.print(
            f"Resuming from checkpoint {path} - Resume step: {global_epoch} - Epoch step: {resume_epoch}"
        )

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = config['time'] * 60
    else:
        time_budget = 2000

    lambda_gp = config['loss']['lambda_gp']
    best_model = False if 'best_model' not in config['train'] else config[
        'train']['best_model']

    latent_dim = config['generator']['latent_dim']
    best_loss = get_best_loss(BASE_DIR)
    for epoch in range(resume_epoch, config['train']['num_epochs']):
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        mean_lossG, mean_lossD = 0, 0
        mean_loss_fake, mean_loss_real, mean_gp = 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            # Discriminator update
            data, label = batch
            label = label.long().to(accelerator.device)
            data = data.float().to(accelerator.device)
            noise_dims = tuple([data.size(0)] + latent_dim)
            noise = torch.randn(*noise_dims).to(accelerator.device)

            fake = netG(noise, label)
            pred_fake = netD(fake, label)
            loss_d_fake = pred_fake.mean()
            pred_real = netD(data, label)
            loss_d_real = pred_real.mean()

            gradient_penalty = calc_gradient_penalty(netD,
                                                     data,
                                                     fake,
                                                     labels=label)

            loss_d = loss_d_fake - loss_d_real + (lambda_gp * gradient_penalty)
            mean_lossD += loss_d.item()
            mean_loss_fake += loss_d_fake.item()
            mean_loss_real += loss_d_real.item()
            mean_gp += gradient_penalty.item()

            optimizerD.zero_grad()
            loss_d.backward()
            optimizerD.step()

            # Generator update
            noise = torch.randn(*noise_dims).to(accelerator.device)
            fake = netG(noise, label)
            pred_fake = netD(fake, label)
            loss_g = -pred_fake.mean()

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            mean_lossG += loss_g.item()
            progress_bar.update(1)
            progress_bar.set_postfix({
                "mean_lossG": mean_lossG / (step + 1),
                "mean_lossD": mean_lossD / (step + 1),
            })

        lr_schedulerD.step()
        lr_schedulerG.step()
        logger.info(
            f"Epoch {epoch} - Mean Loss: {mean_lossD / (step + 1)} - LR: {optimizerD.param_groups[0]['lr']}"
        )
        tlog.write({
            "lossD": mean_lossD / (step + 1),
            "lossG": mean_lossG / (step + 1),
            "loss_real": mean_loss_real / (step + 1),
            "loss_fake": mean_loss_fake / (step + 1),
            "gp": mean_gp / (step + 1)
        })

        if best_model and (best_loss > np.abs(mean_lossD / (step + 1))):
            best_loss = np.abs(mean_lossD / (step + 1))
            best_models = [("generator", accelerator.unwrap_model(netG)),
                           ("discriminator", accelerator.unwrap_model(netD))]
            save_best_model(BASE_DIR, best_models, best_loss)
            logger.info(f"Saved model - Best Loss: {best_loss}")

        global_epoch += 1
        if ((epoch + 1) % config['train']['checkpoint_epoch_freq']) == 0:
            if accelerator.is_main_process:
                save_checkpoint_accelerate(logger, BASE_DIR, config,
                                           accelerator, epoch)
        if ((epoch + 1) % config['samples']['samples_freq']) == 0:
            sampleG = accelerator.unwrap_model(netG)
            sampleG.eval()
            sampling_GAN(config,
                         BASE_DIR,
                         accelerator,
                         sampleG,
                         latent_dim,
                         nsamp=config['samples']['samples_num'],
                         epoch=epoch,
                         conditioned=config['train']['conditional'])

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)

        if (time_budget < np.mean(qe_time)):
            break
    save_checkpoint_accelerate(logger, BASE_DIR, config, accelerator, epoch)
    accelerator.save_state(f'{BASE_DIR}/final/')
    tlog.close()
    logger.info("Saved final model")

    # if best_model:
    #     load_model(netG,
    #                os.path.join(f'{BASE_DIR}/best/', 'generator.safetensors'))

    logger.info(f"Sampling final model {config['train']['conditional']}")
    sampleG = accelerator.unwrap_model(netG)
    sampleG.eval()
    sampling_GAN(config,
                 BASE_DIR,
                 accelerator,
                 sampleG,
                 latent_dim,
                 nsamp=config['samples']['samples_gen'],
                 epoch=epoch,
                 conditioned=config['train']['conditional'],
                 final=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def GANTrain(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])

    cfg['name'] = experiment_name_GAN(cfg)
    main(cfg)


if __name__ == "__main__":
    GANTrain()
