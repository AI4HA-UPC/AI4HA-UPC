import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from accelerate.utils import find_executable_batch_size
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
import logging
from PIL import Image
import os
import torchvision.transforms as T
import argparse

from torchvision.utils import make_grid
from accelerate import DistributedDataParallelKwargs
import wandb
import math
from time import time
import numpy as np
import shutil

from diffusion.models.autoencoder_kl import AutoencoderKL
from diffusion.models.vqvae import VQModel
from diffusion.models.patchdiscriminator import NLayerDiscriminator, weights_init
from diffusion.losses.latent_losses import   DiscriminatorLoss
from diffusion.util import instantiate_from_config, load_config, time_management 


MODELS = {
    "VQVAE": VQModel,
    "AEKL": AutoencoderKL
}

LOSSES = {'L1': F.l1_loss,
          'L2': F.mse_loss}

DIRS = ['checkpoints', 'logs', 'samples']


logger = get_logger(__name__, log_level="INFO")


def training(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    accparams = config['accelerator']
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(
            **config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    logger.info(accelerator.state, main_process_only=False)

    @find_executable_batch_size(starting_batch_size=config['dataset']['dataloader']['batch_size'])
    def train_loop(batch_size):
        logger.info(f"BATCH SIZE = {batch_size}")
        torch.autograd.set_detect_anomaly(True)

        # DATA LOADER
        train_data = instantiate_from_config(config['dataset']['train'])
        test_data = instantiate_from_config(config['dataset']['test'])

        train_dataloader = torch.utils.data.DataLoader(train_data, **config['dataset']["dataloader"])

        # Latent Model
        model = MODELS[config['model']['type']](**config['model']['params'])

        model.config_name = config['name']

        optimizer = torch.optim.AdamW(model.parameters(
        ), lr=config['train']['learning_rate'] * accelerator.num_processes)

        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=config['train']['lr_warmup_steps'] *
            accparams['gradient_accumulation_steps'],
            num_training_steps=(len(train_dataloader) *
                                config['train']['epochs']),
        )

        # Patch GAN
        pgparams = config['discriminator']
        patchgan = NLayerDiscriminator(input_nc=pgparams['channels'],
                                    n_layers=pgparams['nlayers'],
                                    use_actnorm=pgparams['actnorm']
                                    ).apply(weights_init)
        pg_optimizer = torch.optim.AdamW(patchgan.parameters(
        ), lr=config['train']['learning_rate'] * accelerator.num_processes)

        lr_scheduler_pg = get_scheduler(
            "cosine",
            optimizer=pg_optimizer,
            num_warmup_steps=config['train']['lr_warmup_steps'] *
            accparams['gradient_accumulation_steps'],
            num_training_steps=(len(train_dataloader) *
                                config['train']['epochs']),
        )

        # ACCELERATE configuration
        model, patchgan, optimizer, pg_optimizer, train_dataloader, lr_scheduler, lr_scheduler_pg = accelerator.prepare(
            model, patchgan, optimizer, pg_optimizer, train_dataloader, lr_scheduler, lr_scheduler_pg
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_data)}")

        # AE loss
        kl_loss = instantiate_from_config(config['loss']).eval().to(accelerator.device)
           
        disc_loss = DiscriminatorLoss(
            params=config['loss']['params']['params']).to(accelerator.device)

        generator = torch.Generator(accelerator.device)

        # Training params
        total_batch_size = config['dataset']['dataloader']['batch_size'] * \
            accelerator.num_processes * \
            accparams['gradient_accumulation_steps']
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / accparams['gradient_accumulation_steps'])
        max_train_steps = config['train']['epochs'] * \
            num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_data)}")
        logger.info(f"  Num Epochs = {config['train']['epochs']}")
        logger.info(
            f"  Instantaneous batch size per device = {config['dataset']['dataloader']['batch_size']}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        logger.info(f"  LOSS = {config['loss']['class']}")

        global_step = 0
        first_epoch = 0

        # Get the most recent checkpoint
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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(
                f'{BASE_DIR}/checkpoints/', path))

            global_step = int(path.split("_")[1]) #*config['train']['checkpoint_freq']

            resume_global_step = global_step 
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * accparams['gradient_accumulation_steps'])
            accelerator.print(
                f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}")

        hps = {"num_iterations": config['train']['epochs'],
               "learning_rate": config['train']['learning_rate'] * accelerator.num_processes}
        accelerator.init_trackers(config['name'], config=hps, init_kwargs={
                                  "wandb": {"dir": os.path.join(BASE_DIR, "logs")}})

        logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

        last_time = time()
        qe_time = []

        if 'time' in config:
            time_budget = config['time'] * 60
        else:
            time_budget = 2000

        image_key ='image' if "image_key" not in config['model'] else config['model']['image_key']
        for epoch in range(first_epoch, config['train']['epochs']):
            progress_bar = tqdm(total=len(train_dataloader),
                                disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            model.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % accparams['gradient_accumulation_steps'] == 0:
                        progress_bar.update(1)
                    continue

                train_images = batch[image_key].permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                with accelerator.accumulate(model):

                    # VAE/VQVAE (Generator)
                    optimizer.zero_grad()

                    if config['model']['type'] == 'AEKL':
                        output, model_loss = model(train_images, sample_posterior=True, generator=generator)
                    elif config['model']['type'] == 'VQVAE':
                        output, model_loss = model(train_images)
                    else:
                        raise NameError(f"Model {config['model']['type']} not supported")
                    
                    predictions = output.sample
                    logits_fake = patchgan(predictions.contiguous())

                    lossg = kl_loss(train_images, predictions, model_loss, logits_fake,
                                    global_step=global_step,
                                    last_layer=accelerator.unwrap_model(model).get_last_layer())

                    accelerator.backward(lossg)
                    optimizer.step()

                    # PatchGAN (Discriminator)
                    pg_optimizer.zero_grad()

                    logits_real = patchgan(train_images.contiguous().detach())
                    logits_fake = patchgan(predictions.contiguous().detach())

                    lossd = disc_loss(logits_real, logits_fake,
                                      global_step=global_step)

                    accelerator.backward(lossd)
                    pg_optimizer.step()

                    lr_scheduler.step()
                    lr_scheduler_pg.step()

                if accelerator.sync_gradients:
                    progress_bar.update(1)

                    global_step += 1


                    if accelerator.is_local_main_process:
                        accelerator.get_tracker("wandb").log({"lossg": lossg.detach().item(),
                                                            "lossd": lossd.detach().item(), 
                                                            "lr": lr_scheduler.get_last_lr()[0]},
                                                            step=global_step)
                        logs = {"lossg": lossg.detach().item(), 
                                "lossd": lossd.detach().item(), 
                                "lr": lr_scheduler.get_last_lr()[0], 
                                "step": global_step}
                        progress_bar.set_postfix(**logs)
                        if ((global_step + 1) % config['samples']['samples_freq']) == 0:
                            if image_key == 'image':
                                grid = T.ToPILImage(mode='RGB')(make_grid((torch.clamp(predictions, min=-1, max=1)+1)/2, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}s.jpg")
                                grid = T.ToPILImage(mode='RGB')(make_grid((train_images+1)/2, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}r.jpg")
                            elif image_key == 'aesegmentation':
                                # processed = (torch.clamp(torch.unsqueeze(torch.sum(predictions, axis=1),dim=1), min=-1, max=1)+1)/2   
                                processed = (torch.clamp(predictions, min=-1, max=1)+1)/2           
                                # processed = predictions/torch.max(predictions)
                                grid = T.ToPILImage(mode='LA')(make_grid(processed, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}s.png")
                                # processed = (torch.unsqueeze(torch.sum(train_images, axis=1),dim=1)+1)/2
                                processed = (train_images+1)/2
                                # processed = train_images/torch.max(train_images)                         
                                grid = T.ToPILImage(mode='LA')(make_grid(processed, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}r.png")
                            elif image_key == 'imageseg':
                                pred_image = predictions[:, :3, ...]
                                pred_image = (torch.clamp(pred_image, min=-1, max=1)+1)/2
                                grid = T.ToPILImage(mode='RGB')(make_grid(pred_image, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}s.jpg")

                                pred_mask = predictions[:, 3:, ...] 
                                pred_mask = (torch.unsqueeze(torch.sum(pred_mask, axis=1),dim=1)+1)/2
                                grid = T.ToPILImage(mode='LA')(make_grid(pred_mask, nrow=4))
                                grid.save(f"{BASE_DIR}/samples/E{epoch:04d}-S{global_step+1:08d}s.png") 


                        if (global_step+1) % config['train']['checkpoint_freq'] == 0:
                            if accelerator.is_main_process:
                                save_path = os.path.join(
                                    f'{BASE_DIR}/checkpoints/', f"checkpoint_{global_step+1:08d}")
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")
                                dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
                                dirs = sorted(
                                    [d for d in dirs if d.startswith("checkpoint")])
                                if len(dirs) > config["projectconf"]["total_limit"]:
                                    for d in dirs[:-config["projectconf"]["total_limit"]]:
                                        logger.info(
                                            f'delete {BASE_DIR}/checkpoints/{d}')
                                        shutil.rmtree(
                                            f'{BASE_DIR}/checkpoints/{d}')


            last_time, qe_time, time_budget = time_management(last_time, qe_time, time_budget, logger)
            if (time_budget < np.mean(qe_time)):
                break

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            model.save_pretrained(save_directory=BASE_DIR)
        accelerator.end_training()

    train_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()

    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config = load_config(args.config)
    training(config)
