import argparse
import os
from tqdm import tqdm
import numpy as np
import shutil

import torch
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchinfo import summary
import logging
# from tqdm.auto import tqdm
# from Model import WaveGANGenerator, WaveGANDiscriminator
from time import time
from ai4ha.util import instantiate_from_config, load_config_fix_paths, \
    time_management
from ai4ha.GAN.losses import calc_gradient_penalty
from ai4ha.log.textlog import textlog

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from accelerate import DistributedDataParallelKwargs

from safetensors.torch import save_model, load_model

DIRS = ["checkpoints", "logs", "samples", "final", "model", "best"]
logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

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

    # Instantiate the model from the config file classes and parameters
    netG = instantiate_from_config(config['model']['generator'])
    netD = instantiate_from_config(config['model']['discriminator'])

    # Optimizers
    optimizerG = optim.Adam(netG.parameters(),
                            lr=config['optimizer']['lr'],
                            betas=(config['optimizer']['beta1'],
                                   config['optimizer']['beta2']))

    optimizerD = optim.Adam(netD.parameters(),
                            lr=config['optimizer']['lr'],
                            betas=(config['optimizer']['beta1'],
                                   config['optimizer']['beta2']))

    # Load the dataset
    train_data = instantiate_from_config(config['dataset']['train'])
    # test_data = instantiate_from_config(config['dataset']['test'])
    train_dataloader = torch.utils.data.DataLoader(
        train_data, **config['dataset']["dataloader"])

    # Learning rate schedulers
    lr_schedulerD = get_scheduler(
        "cosine",
        optimizer=optimizerD,
        num_warmup_steps=config['train']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) *
                            config['train']['num_epochs']),
    )
    lr_schedulerG = get_scheduler(
        "cosine",
        optimizer=optimizerG,
        num_warmup_steps=config['train']['lr_warmup_steps'] *
        accparams['gradient_accumulation_steps'],
        num_training_steps=(len(train_dataloader) *
                            config['train']['num_epochs']),
    )

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

    global_step = 0

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

    # resume_from_checkpoint = True
    if path is None:
        accelerator.print(
            "Checkpoint does not exist. Starting a new training run.")
        # resume_from_checkpoint = None
        resume_step = 0
    else:
        accelerator.load_state(os.path.join(f'{BASE_DIR}/checkpoints/', path))
        global_step = int(path.split("_")[1]) + 1
        resume_global_step = global_step
        # first_epoch = global_step
        # * accparams['gradient_accumulation_steps']))
        resume_step = resume_global_step
        accelerator.print(
            f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}"
        )

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = config['time'] * 60
    else:
        time_budget = 2000

    lambda_gp = config['train']['lambda_gp']
    best_model = False if 'best_model' not in config['train'] else config[
        'train']['best_model']

    latent_dim = config['model']['generator']['latent_dim']
    best_loss = 1e10
    for epoch in range(resume_step, config['train']['num_epochs']):
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

            lr_schedulerD.step()
            lr_schedulerG.step()
            progress_bar.update(1)
            progress_bar.set_postfix({
                "lossD": loss_d.item(),
                "lossG": loss_g.item(),
                "mean_lossG": mean_lossG / (step + 1),
                "mean_lossD": mean_lossD / (step + 1)
            })

        tlog.write({
            "lossD": mean_lossD / (step + 1),
            "lossG": mean_lossG / (step + 1),
            "loss_real": mean_loss_real / (step + 1),
            "loss_fake": mean_loss_fake / (step + 1),
            "gp": mean_gp / (step + 1)
        })

        if best_model and (best_loss > np.abs(mean_lossD / (step + 1))):
            best_loss = np.abs(mean_lossD / (step + 1))
            generator = accelerator.unwrap_model(netG)
            discriminator = accelerator.unwrap_model(netD)
            save_model(
                generator,
                os.path.join(f'{BASE_DIR}/best/', 'generator.safetensors'))
            save_model(
                discriminator,
                os.path.join(f'{BASE_DIR}/best/', 'discriminator.safetensors'))

            logger.info(f"Saved model - Best Loss: {best_loss}")

        global_step += 1
        if ((epoch + 1) % config['train']['checkpoint_epoch_freq']) == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(f'{BASE_DIR}/checkpoints/',
                                         f"checkpoint_{global_step:06d}")  #
                # save_path = f'{BASE_DIR}/checkpoints/'
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
                dirs = sorted([d for d in dirs if d.startswith("checkpoint")])
                if len(dirs) > config["projectconf"]["total_limit"]:
                    for d in dirs[:-config["projectconf"]["total_limit"]]:
                        logger.info(f'delete {BASE_DIR}/checkpoints/{d}')
                        shutil.rmtree(f'{BASE_DIR}/checkpoints/{d}')

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)

        if (time_budget < np.mean(qe_time)):
            break

    accelerator.save_state(f'{BASE_DIR}/final/')
    # accelerator.save_model(netG, os.path.join(f'{BASE_DIR}/final/',
    #                                           'Generator'))
    # accelerator.save_model(netD,
    #                        os.path.join(f'{BASE_DIR}/final/', 'Discriminator'))
    tlog.close()
    logger.info("Saved final model")

    if best_model:
        load_model(netG,
                   os.path.join(f'{BASE_DIR}/best/', 'generator.safetensors'))

    netG.eval()
    llabels = []
    lsamples = []
    for i in range(config['samples']['samples_gen']):
        noise = torch.randn(*tuple([1] + latent_dim)).to(accelerator.device)
        label = torch.tensor([
            i % config['model']['generator']['params']['n_classes']
        ]).long().to(accelerator.device)
        fake = netG(noise, label)
        lsamples.append(fake.cpu().detach().numpy())
        llabels.append(label.cpu().detach().numpy())
    np.save(f"{BASE_DIR}/samples/samples.npy", np.array(lsamples))
    np.save(f"{BASE_DIR}/samples/labels.npy", np.array(llabels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="configuration file")
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
    # config = load_config(args.config)
    main(config)
