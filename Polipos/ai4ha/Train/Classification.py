import logging
import math
import os
from collections import Counter

import torch

from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

from diffusers.optimization import get_scheduler

from accelerate import DistributedDataParallelKwargs

from time import time
import numpy as np

from ai4ha.util import save_config, time_management
from ai4ha.log.textlog import textlog
from ai4ha.util.train import get_most_recent_checkpoint, \
    save_checkpoint_accelerate, get_optimizer, get_best_loss, \
    save_best_model, load_best_model
from ai4ha.losses import OrdinalLoss, FocalLoss
from ai4ha.util.plot import confusion_matrix_pretty
from ai4ha.util.misc import initialize_dirs
from ai4ha.Classifier import TransformerClassifier
from ai4ha.util.data import load_dataset
from torch.nn.functional import softmax


def validation(network, data, device):
    class_net = network.eval()
    lpred, llabels = [], []
    for iter_idx, (samples, olabels) in enumerate(tqdm(data)):
        samples = samples.type(torch.cuda.FloatTensor).cuda(device,
                                                            non_blocking=True)
        olabels = olabels.cuda(device, non_blocking=True)
        labels = softmax(class_net(samples), dim=1).cpu().detach().numpy()
        labels = np.argmax(labels, -1)
        lpred.extend(list(labels))
        llabels.extend(list(olabels.cpu().detach().numpy()))

    return llabels, lpred


def classifier_loop(config, logger):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    initialize_dirs(
        BASE_DIR, ['checkpoints', 'logs', 'samples', "final", "model", "best"])

    save_config(config, f"{BASE_DIR}/config.yaml")
    accparams = config['accelerator'].copy()
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(
            **config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    logger.info(f"{config['name']}")
    # Prepare the dataset
    # train_data = instantiate_from_config(config['dataset']['train'])
    # test_data = instantiate_from_config(config['dataset']['test'])
    # val_data = instantiate_from_config(config['dataset']['val'])
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                                **config["dataloader"])
    # test_dataloader = torch.utils.data.DataLoader(test_data,
    #                                               **config["dataloader"])
    # val_dataloader = torch.utils.data.DataLoader(val_data,
    #                                              **config["dataloader"])

    dataloaders = load_dataset(config)
    len_train_data = len(dataloaders['train'])
    # logger.info(f"Dataset size: {len(train_data)}")

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
        resume_step = 0
    else:
        global_step = int(
            path.split("_")[1])  #* \ config['train']['checkpoint_freq']
        resume_global_step = global_step  #* accparams['gradient_accumulation_steps']
        first_epoch = global_step  # // num_update_steps_per_epoch
        # * accparams['gradient_accumulation_steps']))
        resume_step = (resume_global_step % (num_update_steps_per_epoch))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    tlog = textlog(f"{BASE_DIR}/logs/losslog.csv",
                   ["CEloss", "ACCTrain", "ACCVal"])

    if config['model']['modeltype'] == "Transformer":
        model = TransformerClassifier(**config['model']['params'])
    else:
        raise ValueError(
            f"Unsupported model type: {config['model']['modeltype']}")

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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_data}")
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
            f"Resuming from checkpoint {path} - Resume step: {global_step} - Epoch step: {resume_step}"
        )

    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    last_time = time()
    qe_time = []

    if 'time' in config:
        time_budget = config['time'] * 60
    else:
        time_budget = 2000

    best_model = False if 'best_model' not in config['train'] else config[
        'train']['best_model']
    best_model_patience = 10 if 'best_model_patience' not in config[
        'train'] else config['train']['best_model_patience']
    # Train!
    if config['loss']['loss'] == "CE":
        cls_criterion = CrossEntropyLoss()
    elif config['loss']['loss'] == "Ordinal":
        cls_criterion = OrdinalLoss(config['model']['params']['n_classes'])
    elif config['loss']['loss'] == "Focal":
        datal = [int(d[1][0]) for d in train_dataloader]
        cnt = Counter(datal)
        prp = torch.tensor([cnt[d] / len(datal)
                            for d in range(config['model']['params']['n_classes'])]).to(accelerator.device)
        cls_criterion = FocalLoss(weight=prp / prp.sum(),
                                  gamma=config['loss']['gamma'])
    else:
        raise ValueError(f"Unsupported loss: {config['loss']['loss']}")

    logger.info(f" LOSS = {config['loss']['loss']}")

    best_loss = get_best_loss(BASE_DIR, min=False)
    for epoch in range(first_epoch, config['train']['num_epochs']):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            samples, labels = batch
            samples = samples.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.long)
            with accelerator.accumulate(model):
                model.zero_grad()
                r_out_cls = model(samples)
                loss = cls_criterion(r_out_cls, labels)
                mean_loss += loss.item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "m_loss": mean_loss / (step + 1),
                "step": global_step
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

        lr_scheduler.step()
        logger.info(
            f"Epoch {epoch} - Mean Loss: {mean_loss / (step + 1)} - LR: {optimizer.param_groups[0]['lr']}"
        )

        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            llabels, lpred = validation(model, dataloaders['train'],
                                        accelerator.device)

            vllabels, vlpred = validation(model, dataloaders['val'],
                                          accelerator.device)

            acctrain = accuracy_score(llabels, lpred)
            accval = accuracy_score(vllabels, vlpred)
            tlog.write({
                "CEloss": mean_loss / (step + 1),
                "AccTrain": acctrain,
                "AccVal": accval
            })
            logger.info(
                f"ACC train= {acctrain} - val= {accval} - step= {global_step}")

            if best_model and (epoch > best_model_patience) and (accval
                                                                 > best_loss):
                save_best_model(
                    BASE_DIR,
                    [('classifier', accelerator.unwrap_model(model))], accval)
                best_loss = accval
                logger.info(f"Saved model - Best Loss: {best_loss}")

            if ((epoch + 1) % config['train']['checkpoint_epoch_freq']
                    == 0) or epoch == (config['train']['num_epochs'] - 1):
                # save the model

                save_checkpoint_accelerate(logger, BASE_DIR, config,
                                           accelerator, epoch)

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)
        if (time_budget < np.mean(qe_time)):
            break

    save_checkpoint_accelerate(logger, BASE_DIR, config, accelerator,
                               config['train']['num_epochs'])
    tlog.close()
    logger.info("Finishing training")
    accelerator.end_training()

    if best_model:
        model = accelerator.unwrap_model(model)
        model = load_best_model(BASE_DIR,
                                [('classifier', model)])['classifier']
    else:
        model = accelerator.unwrap_model(model)

    llabels, lpred = validation(model, dataloaders['val'], accelerator.device)
    logger.info("*** Validation Results ***")
    val_acc = accuracy_score(llabels, lpred)
    logger.info(f"ACC = {val_acc}")

    logger.info(f'\n{classification_report(llabels, lpred)}')
    logger.info(
        f'\n{confusion_matrix_pretty(llabels, lpred, [str(l) for l in range(config["dataset"]["nclasses"])])}'
    )

    if dataloaders['test'] is not None:
        llabels, lpred = validation(model, dataloaders['test'],
                                    accelerator.device)
        logger.info("*** Test Results ***")
        test_acc = accuracy_score(llabels, lpred)
        logger.info(f"ACC = {test_acc}")

        logger.info(f'\n{classification_report(llabels, lpred)}')
        logger.info(
            f'\n{confusion_matrix_pretty(llabels, lpred, [str(l) for l in range(config["dataset"]["nclasses"])])}'
        )

    return val_acc
