import torch
import os
import logging
import shutil
import numpy as np
from tqdm.auto import tqdm
import argparse
from time import time
from ai4ha.util import instantiate_from_config, load_config, time_management
from ai4ha.log.textlog import textlog
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from ai4ha.Autoencoders import ExtraMAE
from ai4ha.layers import PatchDiscriminator1D
import math

from accelerate import DistributedDataParallelKwargs

from diffusers.optimization import get_scheduler

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
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
                   ["MSEloss", "advloss", "advlossF"])

    model = ExtraMAE(
        in_channels=config["model"]["in_channels"],
        series_length=config["model"]["series_length"],
        mask_percent=config["model"]["mask_percent"],
        num_layers=config["model"]["layers"],
        num_heads=config["model"]["heads"],
        embed_dimension=config["model"]["embed_dim"],
        patch_size=config["model"]["patch_size"],
    )

    adversarial = PatchDiscriminator1D(
        sequence_length=config["model"]["series_length"],
        in_channels=config["model"]["in_channels"],
        embed_dimension=config["model"]["embed_dim"],
        patch_size=config["model"]["patch_size"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"] * accelerator.num_processes,
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        weight_decay=config["optimizer"]["weight_decay"],
        eps=config["optimizer"]["epsilon"],
    )

    optimizerA = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"] * accelerator.num_processes,
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        weight_decay=config["optimizer"]["weight_decay"],
        eps=config["optimizer"]["epsilon"],
    )

    train_data = instantiate_from_config(config["dataset"]["train"])
    # test_data = instantiate_from_config(config["dataset"]["test"])
    train_dataloader = torch.utils.data.DataLoader(
        train_data, **config["dataset"]["dataloader"])

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["train"]["lr_warmup_steps"] *
        accparams["gradient_accumulation_steps"],
        num_training_steps=(len(train_dataloader) * config["train"]["epochs"]),
    )

    model, adversarial, optimizer, optimizerA, train_dataloader, lr_scheduler = accelerator.prepare(
        model, adversarial, optimizer, optimizerA, train_dataloader,
        lr_scheduler)

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
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    model.to(accelerator.device)

    num_update_steps_per_epoch = (
        len(train_dataloader) //
        config["accelerator"]["gradient_accumulation_steps"])

    global_step = 0

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
        global_step = int(path.split("_")[1]) + 1
        resume_global_step = global_step
        first_epoch = global_step
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

    model.train()
    for epoch in range(first_epoch, config["train"]["epochs"]):
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        mean_advloss = 0
        mean_advlossF = 0
        for step, batch in enumerate(train_dataloader):
            # if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % accparams['gradient_accumulation_steps'] == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(model):
                data, label = batch
                data = data.float().to(accelerator.device)
                pred, mask = model(data)
                batch_mask = data * mask
                pred_mask = pred * mask

                rloss = torch.log(adversarial(batch_mask))
                ploss = torch.log(1 - adversarial(pred_mask))
                advloss = (rloss + ploss).mean()
                mean_advloss += advloss.item()

                accelerator.backward(advloss)
                optimizerA.step()
                optimizerA.zero_grad()

                pred, mask = model(data)
                batch_mask = data * mask
                pred_mask = pred * mask
                ploss = torch.log(adversarial(pred_mask))
                advlossF = ploss.mean()
                mean_advlossF += advlossF.item()

                loss = torch.nn.functional.mse_loss(pred_mask,
                                                    batch_mask,
                                                    reduction="none")
                loss = loss.mean() + 0.5 * advlossF
                mean_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "mean_loss": mean_loss / (step + 1),
                    "advlossR": mean_advloss / (step + 1),
                    "advlossF": mean_advlossF / (step + 1)
                })

        tlog.write({
            "MSEloss": mean_loss / (step + 1),
            "advlossR": mean_advloss / (step + 1),
            "advlossF": mean_advlossF / (step + 1)
        })

        global_step += 1
        if (epoch % config['train']['checkpoint_epoch_freq'] + 1) == 0:
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

    tlog.close()
    # accelerator.end_training()
    save_path = os.path.join(f'{BASE_DIR}/model/', "model")
    accelerator.save_state(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="configuration file")

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
