# from cv2 import log
import torch
import os
import logging
import numpy as np
from tqdm.auto import tqdm
from time import time
import math
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
from accelerate import DistributedDataParallelKwargs

from ai4ha.util import instantiate_from_config, time_management, \
        fix_paths, save_config, experiment_name_autoencoder
from ai4ha.log.textlog import textlog
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from ai4ha.Autoencoders import ExtraMAE
from ai4ha.util.train import get_most_recent_checkpoint, \
    save_checkpoint_accelerate, get_optimizer, get_lr_scheduler
from ai4ha.util.sampling import sampling_MAE

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

    save_config(config, f"{BASE_DIR}/config.yaml")
    accparams = config["accelerator"]
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
    tlog = textlog(f"{BASE_DIR}/logs/losslog.csv", ["MSEloss"])

    train_data = instantiate_from_config(config["dataset"]["train"])
    # test_data = instantiate_from_config(config["dataset"]["test"])
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   **config["dataloader"])
    global_step = 0
    # Get the most recent checkpoint
    path = get_most_recent_checkpoint(logger, BASE_DIR)

    if path is not None:
        global_epoch = int(path.split("_")[1]) + 1
        resume_epoch = global_epoch
    else:
        global_epoch = 0
        resume_epoch = 0

    model = ExtraMAE(
        in_channels=config["model"]["in_channels"],
        series_length=config["model"]["series_length"],
        mask_percent=config["model"]["mask_percent"],
        num_layers_encoder=config["model"]["layers_encoder"],
        num_heads_encoder=config["model"]["heads_encoder"],
        num_layers_decoder=config["model"]["layers_decoder"],
        num_heads_decoder=config["model"]["heads_decoder"],
        embed_dimension=config["model"]["embed_dim"],
        patch_size=config["model"]["patch_size"],
        norm_layer=config["model"]["norm_layer"],
        patcher=config["model"]["patcher"],
    )

    # Compute the loss with only the masked values
    if 'mask_loss' in config['model']:
        mask_loss = config['model']['mask_loss']
    else:
        mask_loss = True

    # Initialize the optimizer
    optimizer = get_optimizer(model, accelerator, config)

    logger.info(f"* LR scheduler= {config['lr_scheduler']['type']}")
    lr_scheduler = get_lr_scheduler(config, optimizer, train_dataloader,
                                    accparams)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    model.to(accelerator.device)

    num_update_steps_per_epoch = len(train_dataloader) // config[
        'accelerator']['gradient_accumulation_steps']

    total_batch_size = config['dataloader']['batch_size'] * \
        accelerator.num_processes * accparams['gradient_accumulation_steps']
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / accparams['gradient_accumulation_steps'])
    max_train_steps = config['train']['num_epochs'] * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
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

    model.train()
    if config['loss']['loss'] == 'L2':
        lossf = torch.nn.functional.mse_loss
    elif config['loss']['loss'] == 'L1':
        lossf = torch.nn.functional.l1_loss
    else:
        raise ValueError(f"Loss {config['loss']['loss']} not supported")

    for epoch in range(resume_epoch, config["train"]["num_epochs"]):
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                data, label = batch
                data = data.float().to(accelerator.device)
                pred, mask = model(data)

                if mask_loss:
                    batch_mask = data * mask
                    pred_mask = pred * mask

                loss = lossf(pred_mask, batch_mask, reduction="none")
                loss = loss.mean()
                mean_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()

                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "mean_loss": mean_loss / (step + 1),
                })

        lr_scheduler.step(mean_loss / (step + 1))
        logger.info(
            f"Epoch {epoch} - Mean Loss: {mean_loss / (step + 1)} - LR: {optimizer.param_groups[0]['lr']}"
        )
        tlog.write({
            "MSEloss": mean_loss / (step + 1),
        })

        global_step += 1
        if ((epoch + 1) % config['train']['checkpoint_freq']) == 0:
            if accelerator.is_main_process:
                save_checkpoint_accelerate(logger, BASE_DIR, config,
                                           accelerator, epoch)

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)

        if (time_budget < np.mean(qe_time)):
            break

    save_checkpoint_accelerate(logger, BASE_DIR, config, accelerator, epoch)

    accelerator.save_state(f'{BASE_DIR}/final/')
    logger.info("Saved final model")
    tlog.close()

    sampling_MAE(BASE_DIR,
                 accelerator,
                 model,
                 train_data,
                 nsamp=config['samples']['samples_gen'])


@hydra.main(version_base=None, config_path="conf", config_name="config")
def extraMAETrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    cfg['name'] = experiment_name_autoencoder(cfg)
    main(cfg)


if __name__ == "__main__":
    extraMAETrain()
