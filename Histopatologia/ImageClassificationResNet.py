"""
Training from scratch a ResNet model using Transformers package of Hugging Face
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datasets
from datasets import Dataset, Image, load_metric
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, ViTImageProcessor, SwinForImageClassification, SchedulerType, get_scheduler, ResNetForImageClassification
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch.nn as nn

from diffusion.util import load_config


DIRS = ['checkpoints','logs', "model"]

logger = get_logger(__name__)


def main(config):

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)  

    accparams = config['accelerator']
    # accparams["logging_dir"] = f"{BASE_DIR}/logs"
    accparams["project_dir"] = BASE_DIR

    if 'projectconf' in config:
        accparams['project_config'] = ProjectConfiguration(**config['projectconf'])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=accparams['gradient_accumulation_steps'] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config['seed'] is not None:
        set_seed(config['seed'])

    # Import data
    train_df = pd.read_csv(config['dataset']['train']['csv_path'])
    test_df = pd.read_csv(config['dataset']['test']['csv_path'])

    train_image_list = train_df['Image_id'].tolist()
    train_image_list = [os.path.join(config['dataset']['train']['data_path'], 'patches/'+img)+'.png' for img in train_image_list]

    test_image_list = test_df['Image_id'].tolist()
    test_image_list = [os.path.join(config['dataset']['test']['data_path'], 'patches/'+img)+'.png' for img in test_image_list]

    train_label_list = train_df['Gleason'].tolist()
    test_label_list = test_df['Gleason'].tolist()

    train_dataset = Dataset.from_dict({"image": train_image_list, "label": train_label_list}).cast_column("image", Image())
    test_dataset = Dataset.from_dict({"image": test_image_list, "label": test_label_list}).cast_column("image", Image())

    # If we don't have a validation split, split off a percentage of train as validation.
    if 'train_val_split' in config['dataset']:
        if isinstance(config['dataset']['train_val_split'], float) and config['dataset']['train_val_split'] > 0.0:
            splits = train_dataset.train_test_split(test_size=config['dataset']['train_val_split'])
            train_dataset = splits['train']
            val_dataset = splits['test']

    # Prepare label mappings.
    id2label = config["dataset"]["id2label"]
    label2id = {v: k for k, v in id2label.items()} 

    image_processor = ViTImageProcessor.from_pretrained(config['model']['model_path'])

    # Load pretrained model and image processor
    config_model = AutoConfig.from_pretrained(config['model']['model_path'])
    config_model.id2label = id2label
    config_model.label2id = label2id

    # Import the model using the config (if trained from scratch) or from_pretrained (if fine-tune)
    if config['train']['pretrain']:
        logger.info("Loading model...")
        model = ResNetForImageClassification.from_pretrained(
            config['model']['model_path'],
            num_labels=config['model']['num_classes'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        if "hidden_dropout_prob" in config["model"]:
            config_model.hidden_dropout_prob = config["model"]["hidden_dropout_prob"]
            model = ResNetForImageClassification(config_model)
        else:
            model = ResNetForImageClassification(config_model)

    # Replace the classifier
    #num_classes = config['model']['num_classes']
    #model.classifier = nn.Linear(config_model.hidden_size, num_classes)

    # Initialize the new layer
    #nn.init.xavier_uniform_(model.classifier.weight)
    #nn.init.zeros_(model.classifier.bias)

    # Preprocessing the datasets

    # Define torchvision transforms to be applied to each image.
    image_mean, image_std = image_processor.image_mean, image_processor.image_std
    size = image_processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                #RandomResizedCrop(size),
                Resize((size, size)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize((size, size)),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    #with accelerator.main_process_first():
    if 'max_train_samples' in config["dataset"]:
        train_dataset = train_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_train_samples']))
    # Set the training transforms
    #train_dataset = train_dataset.set_transform(train_transforms)
    train_dataset.set_transform(train_transforms)
    if 'max_eval_samples' in config["dataset"]:
        val_dataset = val_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_eval_samples']))
    # Set the validation transforms
    #val_dataset = val_dataset.set_transform(val_transforms)
    val_dataset.set_transform(val_transforms)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size']
    )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size'])

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["optimizer"]["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["train"]["learning_rate"])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accparams["gradient_accumulation_steps"])
    if 'max_train_steps' not in config["train"]:
        max_train_steps = config["train"]["epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        max_train_steps = config['train']['max_train_steps']

    lr_scheduler = get_scheduler(
        name=config["train"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["train"]["lr_warmup_steps"] * accparams["gradient_accumulation_steps"],
        num_training_steps=max_train_steps * accparams["gradient_accumulation_steps"],
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accparams["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        max_train_steps = config["train"]["epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config["train"]["epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = config["train"]["checkpoint_epoch_freq"]
    if checkpointing_steps is not None: #and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)  

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #if args.with_tracking:
    #    experiment_config = vars(args)
    #    # TensorBoard cannot log Enums, need the raw value
    #    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #    accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Get the metric function
    metric_acc = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/accuracy.py", experiment_id=config['name'])
    metric_prec = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/precision.py", experiment_id=config['name'])
    metric_recall = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/recall.py", experiment_id=config['name'])
    metric_f1 = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/f1.py", experiment_id=config['name'])

    # Train!
    total_batch_size = config['dataset']["dataloader"]["batch_size"] * accelerator.num_processes * accparams["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['train']['epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['dataset']['dataloader']['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    # Get the most recent checkpoint
    dirs = os.listdir(f'{BASE_DIR}/checkpoints/')
    dirs = [d for d in dirs if d.startswith("epoch")]
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
        resume_step = 0
    else:
        checkpoint_path = os.path.join(f'{BASE_DIR}/checkpoints/', path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * accparams["gradient_accumulation_steps"]
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // accparams["gradient_accumulation_steps"]
            resume_step -= starting_epoch * len(train_dataloader)

    """
    if resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * accparams["gradient_accumulation_steps"]
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // accparams["gradient_accumulation_steps"]
            resume_step -= starting_epoch * len(train_dataloader)
    """

    hps = {"num_iterations": config['train']['epochs'], "learning_rate": config['train']['learning_rate'] * accelerator.num_processes}
    accelerator.init_trackers(config['name'], config=hps, 
                              init_kwargs={"wandb":
                                           {"dir":os.path.join(BASE_DIR, "logs")}})
    with_tracking = True   

    checkpoints_dir = f"{BASE_DIR}/checkpoints"
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    train_loss_list = []
    val_loss_list = []
    num_of_epochs = 0
    for epoch in range(starting_epoch, config["train"]["epochs"]):
        num_of_epochs += 1

        model.train()
        if with_tracking:
            total_loss = 0
            total_loss_val = 0
        if resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if epoch % checkpointing_steps == 0:
                output_dir = f"epoch_{epoch}"
                output_dir = os.path.join(checkpoints_dir, output_dir)
                accelerator.save_state(output_dir)

                if config["push_to_hub"] and epoch < config["train"]["epochs"] - 1:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        checkpoints_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        image_processor.save_pretrained(checkpoints_dir)
                        repo.push_to_hub(
                            commit_message=f"Training in progress {epoch} epochs",
                            blocking=False,
                            auto_lfs_prune=True,
                        )

        #if completed_steps >= max_train_steps:
        #    break

        model.eval()
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss_val = outputs.loss
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss_val += loss_val.detach().float()
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

            metric_acc.add_batch(
                predictions=predictions,
                references=references,
            )
            metric_prec.add_batch(
                predictions=predictions,
                references=references,
            )
            metric_recall.add_batch(
                predictions=predictions,
                references=references,
            )
            metric_f1.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric_acc = metric_acc.compute()
        eval_metric_prec = metric_prec.compute(average='macro')
        eval_metric_recall = metric_recall.compute(average='macro')
        eval_metric_f1 = metric_f1.compute(average='macro')
        logger.info(f"epoch {epoch} = Loss: {total_loss}, Lr: {lr_scheduler.get_last_lr()[0]}, Accuracy: {list(eval_metric_acc.values())[0]}, Precision: {list(eval_metric_prec.values())[0]}, Recall: {list(eval_metric_recall.values())[0]}, F1: {list(eval_metric_f1.values())[0]}")

        if with_tracking:
            accelerator.log(
                {
                    "accuracy": list(eval_metric_acc.values())[0],
                    "precision": list(eval_metric_prec.values())[0],
                    "recall": list(eval_metric_recall.values())[0],
                    "f1": list(eval_metric_f1.values())[0],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if config["push_to_hub"] and epoch < config["train"]["epochs"] - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoints_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(checkpoints_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        # Save losses 
        train_loss_list.append(total_loss)
        val_loss_list.append(total_loss_val)

        #output_dir = f"epoch_{epoch}"
        #output_dir = os.path.join(checkpoints_dir, output_dir)
        #accelerator.save_state(output_dir)

    if with_tracking:
        accelerator.end_training()

    out_dir = f"{BASE_DIR}/model"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        out_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    if accelerator.is_main_process:
        image_processor.save_pretrained(out_dir)
        if config["push_to_hub"]:
            repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        #all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        all_results = {
            "accuracy": list(eval_metric_acc.values())[0],
            "precision": list(eval_metric_prec.values())[0],
            "recall": list(eval_metric_recall.values())[0],
            "f1": list(eval_metric_f1.values())[0],
        }
        with open(os.path.join(out_dir, "validation_results.json"), "w") as f:
            json.dump(all_results, f)

    # Save loss 
    print(train_loss_list)
    train_loss_list_cpu = [x.cpu().numpy() if torch.is_tensor(x) else x for x in train_loss_list]
    val_loss_list_cpu = [x.cpu().numpy() if torch.is_tensor(x) else x for x in val_loss_list]
    print(train_loss_list_cpu)

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, num_of_epochs+1)), train_loss_list_cpu, 'bo-', label='Training loss')
    plt.plot(list(range(1, num_of_epochs+1)), val_loss_list_cpu, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_dir + '/train_val_loss.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)