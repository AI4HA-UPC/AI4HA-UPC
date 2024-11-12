import torch
import os
import logging
import shutil
import numpy as np
from tqdm.auto import tqdm
import argparse
from time import time
from util import instantiate_from_config, load_config, time_management, \
        load_config_fix_paths
from log.textlog import textlog
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from models.ExtraMAE import ExtraMAE
import math
from data_preprocess import data_preprocess

from accelerate import DistributedDataParallelKwargs

from diffusers.optimization import get_scheduler
from ExtraMAEDataset import ExtraMAEDataset
import pickle
import collections

DIRS = ["checkpoints", "logs", "samples", "final", "model"]
logger = get_logger(__name__, log_level="INFO")


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)

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

    model = ExtraMAE(
        in_channels=config["model"]["in_channels"],
        series_length=config["model"]["series_length"],
        mask_percent=config["model"]["mask_percent"],
        num_layers=config["model"]["layers"],
        num_heads=config["model"]["heads"],
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
    
    real_data = f'{config["dataset"]["train_file"]}/train_data.pickle'
    with open(real_data, "rb") as real_file:
    	trainX = np.squeeze(pickle.load(real_file))
    	
    real_labels = f'{config["dataset"]["train_file"]}/train_labels.pickle'
    with open(real_labels, "rb") as real_file:
    	trainY = np.squeeze(pickle.load(real_file))
    	
    	
    dataset = ExtraMAEDataset(trainX, None, trainY)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, **config['dataset']["dataloader"])
            
    train_data = trainX
            
    print(trainX.shape, trainY.shape)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["train"]["lr_warmup_steps"] *
        accparams["gradient_accumulation_steps"],
        num_training_steps=(len(train_dataloader) * config["train"]["epochs"]),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

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
        
    #Train

    model.train()
    for epoch in range(first_epoch, config["train"]["epochs"]):
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            # if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % accparams['gradient_accumulation_steps'] == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(model):
                data, _, _, label = batch
                data = data.permute(0, 2, 1)
                data = data.float().to(accelerator.device)
                pred, mask = model(data)
                data = data[:, :, :mask.shape[2]]
                
                loss = torch.nn.functional.mse_loss(pred,
                                                    data,
                                                    reduction="none")
                loss = loss.mean()
                mean_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "mean_loss": mean_loss / (step + 1),
                })

        tlog.write({
            "MSEloss": mean_loss / (step + 1),
        })

        global_step += 1
        if (epoch % config['train']['checkpoint_epoch_freq'] + 1) == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(f'{BASE_DIR}/checkpoints/',
                                         f"checkpoint_{global_step:06d}")  
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
    save_path = os.path.join(f'{BASE_DIR}/model/', "model")
    accelerator.save_state(save_path)
    
    # Generate!
    masks = []
    results = []
    orig = []
    labels = []
    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )

    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            data, _, _, label = batch
            data = data.permute(0, 2, 1)
            res, mask = model(data)
            for serie, lb in zip(data, label):
            	shape = (5,)
            	mask = torch.randint(0, 2, shape, dtype=torch.uint8)
            	num_ones = int(config["model"]["mask_percent"] * mask.numel())  
            	values = torch.cat((torch.ones(num_ones, dtype=torch.uint8), torch.zeros(mask.numel() - num_ones, dtype=torch.uint8)))
            	mask = values[torch.randperm(mask.numel())]  # Shuffle to distribute 1s and 0s randomly
            	data = data.float().to(accelerator.device)
            	res = model.impute_one(data, mask)
            	masks.append(mask.detach().cpu().numpy())
            	results.append(res.detach().cpu().numpy())
            	orig.append(serie.detach().cpu().numpy())
            	labels.append(lb.detach().cpu().numpy())
            	
    with open(f"{BASE_DIR}/originals/originals_{config['dataset']['store_name']}.pickle", "wb") as fb:
            pickle.dump(orig, fb)    
    with open(f"{BASE_DIR}/samples/samples_{config['dataset']['store_name']}.pickle", "wb") as fb:
            pickle.dump(results, fb)
    with open(f"{BASE_DIR}/masks/masks_{config['dataset']['store_name']}.pickle", "wb") as fb:
            pickle.dump(masks, fb)
            
    with open(f"{BASE_DIR}/labels/labels_{config['dataset']['store_name']}.pickle", "wb") as fb:
            pickle.dump(labels, fb)
            

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
    main(config)

