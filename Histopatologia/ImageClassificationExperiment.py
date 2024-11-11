import argparse
import json
import logging
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

import evaluate
import datasets
from datasets import Dataset, Image, load_metric, concatenate_datasets
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

from diffusion.ols.online_label_smooth import OnlineLabelSmoothing
from diffusion.normalization.reinhard import NormalizerTransform

import transformers
from transformers import AutoConfig, ViTImageProcessor, SwinForImageClassification, SchedulerType, get_scheduler, ResNetForImageClassification, ConvNextForImageClassification, EfficientNetForImageClassification, ConvNextV2ForImageClassification
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch.nn as nn

from diffusion.util import load_config
from diffusion.losses.WeightedKappaLoss import WeightedKappaLoss
from diffusion.losses.focal_loss import FocalLoss


DIRS = ['checkpoints','logs', "model"]

MODELS = {
    'swinv2': SwinForImageClassification,
    'resnet': ResNetForImageClassification,
    'convnext': ConvNextForImageClassification,
    'efficientnet': EfficientNetForImageClassification,
    'convnext2': ConvNextV2ForImageClassification
}

logger = get_logger(__name__)


def main(config):

    # If passed along, set the training seed now.
    if config['seed'] is not None:
        set_seed(config['seed'])

    # Import real and synthetic data
    label = config["model"]["label"]
    real_train_df = pd.read_csv(config['dataset']['real']['csv_path'])
    gen_train_df = pd.read_csv(config['dataset']['generated']['csv_path'])

    real_train_image_list = real_train_df['Image_id'].tolist()
    real_train_image_list = [os.path.join(config['dataset']['real']['data_path'], 'patches/'+img)+'.png' for img in real_train_image_list]

    gen_train_image_list = gen_train_df['Image_id'].tolist()
    gen_train_image_list = [os.path.join(config['dataset']['generated']['data_path'], 'patches/'+img)+'.png' for img in gen_train_image_list]

    real_train_label_list = real_train_df[label].tolist()
    gen_train_label_list = gen_train_df[label].tolist()

    real_train_dataset = Dataset.from_dict({"image": real_train_image_list, "label": real_train_label_list}).cast_column("image", Image())
    gen_train_dataset = Dataset.from_dict({"image": gen_train_image_list, "label": gen_train_label_list}).cast_column("image", Image())

    # If we don't have a validation split, split off a percentage of train as validation.
    if 'train_val_split' in config['dataset']:
        if isinstance(config['dataset']['train_val_split'], float) and config['dataset']['train_val_split'] > 0.0:
            splits = real_train_dataset.train_test_split(test_size=config['dataset']['train_val_split'])
            real_train_dataset = splits['train']
            real_val_dataset = splits['test']

    # Create the subset of the training data 
    random.seed(10)
    idxs = random.sample(range(len(real_train_dataset)), k=config['experiment']['number_of_samples'])
    real_train_dataset_subset = real_train_dataset.select(idxs)

    print(f"Reduced dataset (original dataset) train: {len(real_train_dataset_subset)}") 
    print(f"Reduced dataset (original dataset) validation: {len(real_val_dataset)}")
    print(f"Generated dataset train: {len(gen_train_dataset)}")

    # Prepare label mappings.
    id2label = config["dataset"]["id2label"]
    label2id = {v: k for k, v in id2label.items()} 

    # Process the data
    image_processor = ViTImageProcessor.from_pretrained(config['model']['model_path'])

    # Load pretrained model and image processor
    config_model = AutoConfig.from_pretrained(config['model']['model_path'])
    config_model.id2label = id2label
    config_model.label2id = label2id

    # Import the model using the config (if trained from scratch) or from_pretrained (if fine-tune)
    if config['train']['pretrain']:
        logger.info("Loading model...")
        model = MODELS[config['model']['model_name']].from_pretrained(
            config['model']['model_path'],
            num_labels=config['model']['num_classes'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        if "hidden_dropout_prob" in config["model"]:
            config_model.hidden_dropout_prob = config["model"]["hidden_dropout_prob"]
            model = MODELS[config['model']['model_name']](config_model)
        else:
            model = MODELS[config['model']['model_name']](config_model)
    
    # Replace the classifier if needed 
    try:
        new_classifier = model.classifier.out_features != len(id2label)
    except AttributeError:
        new_classifier = model.classifier[-1].out_features != len(id2label)
    
    if new_classifier:
        num_classes = config['model']['num_classes']
        model.classifier = nn.Linear(config_model.hidden_size, num_classes)

        # Initialize the new layer
        nn.init.xavier_uniform_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)
    
    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    image_mean, image_std = image_processor.image_mean, image_processor.image_std
    if config['model']['model_name'] == 'convnext2':
        size = image_processor.size["shortest_edge"]
    else:
        size = image_processor.size["height"]

    stain_normalization = NormalizerTransform()
    stain_normalization.fit(image_path='/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/normalization/reference_img.png')
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                #RandomResizedCrop(size),
                Resize((size, size)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                stain_normalization,
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize((size, size)),
                CenterCrop(size),
                stain_normalization,
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

    if 'max_train_samples' in config["dataset"]:
        real_train_dataset_subset = real_train_dataset_subset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_train_samples']))
    # Set the training transforms
    real_train_dataset_subset.set_transform(train_transforms)
    gen_train_dataset.set_transform(train_transforms)

    if 'max_eval_samples' in config["dataset"]:
        real_val_dataset = real_val_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_eval_samples']))
    # Set the validation transforms
    real_val_dataset.set_transform(val_transforms)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Label Smoothing and loss
    if config['train']['ols']:
        label_smoothing = True
    else:
        label_smoothing = False
    
    if label_smoothing:
        loss_fn = OnlineLabelSmoothing(alpha=0.5,
                                    n_classes=config['model']['num_classes'],
                                    smoothing=0.1).to('cuda')
    else:
        if config['train']['loss'] == 'crossentropy':
            loss_fn = nn.CrossEntropyLoss()
        elif config['train']['loss'] == 'weightedkappa':
            loss_fn = WeightedKappaLoss(
                num_classes=12,
                mode='quadratic'
            )
        elif config['train']['loss'] == 'focal':
            loss_fn = FocalLoss()
    
    # Get the metrics
    metric_acc = evaluate.load("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/accuracy.py")
    metric_prec = evaluate.load("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/precision.py")
    metric_recall = evaluate.load("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/recall.py")
    metric_f1 = evaluate.load("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/f1.py")

    # Define the experiments 
    proportions_real = config['experiment']['real']
    proportions_generated = config['experiment']['generated']
    number_of_samples = config['experiment']['number_of_samples']

    train_acc = []
    val_acc = []
    train_pre = []
    val_pre = []
    train_rec = []
    val_rec = []
    train_f1 = []
    val_f1 = []
    for xs in itertools.product(proportions_real, proportions_generated):
        print(f"Real: {xs[0]*100}% - Generated: {xs[1]*100}%") 

        # Define the final datasets
        if xs[0] > 0:
            new_idxs = random.sample(range(len(real_train_dataset_subset)), k=int(number_of_samples*xs[0]))
            final_real_train_dataset = real_train_dataset_subset.select(new_idxs)

            if xs[1] > 0:
                idxs_gen = random.sample(range(len(gen_train_dataset)), k=int(number_of_samples*xs[1]))
                final_gen_train_dataset = gen_train_dataset.select(idxs_gen)
            
                final_train_dataset = concatenate_datasets([final_real_train_dataset, final_gen_train_dataset])
            else:
                final_train_dataset = final_real_train_dataset
        
        if xs[0] == 0:
            if xs[1] == 0:
                continue
            else:
                idxs_gen = random.sample(range(len(gen_train_dataset)), k=int(number_of_samples*xs[1]))
                final_gen_train_dataset = gen_train_dataset.select(idxs_gen)

                final_train_dataset = final_gen_train_dataset 

        final_train_dataloader = DataLoader(
            final_train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size']
        )
        real_val_dataloader = DataLoader(real_val_dataset, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size'])

        # Define the base dirctory 
        BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}/Real{int(xs[0]*100)}-Generated{int(xs[1]*100)}"

        for dir in DIRS:
            os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)  

        accparams = config['accelerator']
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
        num_update_steps_per_epoch = math.ceil(len(final_train_dataset) / accparams["gradient_accumulation_steps"])
        if 'max_train_steps' not in config["train"]:
            max_train_steps = config["train"]["epochs"] * num_update_steps_per_epoch
            overrode_max_train_steps = True
        else:
            max_train_steps = config['train']['max_train_steps']

        if config["train"]["lr_warmup_steps"] == 0:
            lr_warmup = None 
        else:
            lr_warmup = config["train"]["lr_warmup_steps"] * accparams["gradient_accumulation_steps"]
        lr_scheduler = get_scheduler(
            name=config["train"]["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=lr_warmup,
            num_training_steps=max_train_steps * accparams["gradient_accumulation_steps"],
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, final_train_dataloader, real_val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, final_train_dataloader, real_val_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(final_train_dataloader) / accparams["gradient_accumulation_steps"])
        if overrode_max_train_steps:
            max_train_steps = config["train"]["epochs"] * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        config["train"]["epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = config["train"]["checkpoint_epoch_freq"]
        if checkpointing_steps is not None: #and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)  

        # Train!
        total_batch_size = config['dataset']["dataloader"]["batch_size"] * accelerator.num_processes * accparams["gradient_accumulation_steps"]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(final_train_dataset)}")
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
                starting_epoch = resume_step // len(final_train_dataloader)
                completed_steps = resume_step // accparams["gradient_accumulation_steps"]
                resume_step -= starting_epoch * len(final_train_dataloader)

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
        if label_smoothing:
            loss_fn.train()
        
        for epoch in range(starting_epoch, config["train"]["epochs"]):
            num_of_epochs += 1

            model.train()
            if with_tracking:
                total_loss = 0
                total_loss_val = 0
                total_loss_train = 0
            if resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(final_train_dataloader, resume_step)
            else:
                active_dataloader = final_train_dataloader

            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = loss_fn(outputs.logits, batch['labels'].to('cuda'))
                    #loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()

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

            model.eval()
            if label_smoothing:
                loss_fn.eval()
            
            val_predictions = []
            val_labels = []
            for step, batch in enumerate(real_val_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss_val = loss_fn(outputs.logits, batch['labels'].to('cuda'))
                    # We keep track of the loss at each epoch
                    if with_tracking:
                        total_loss_val += loss_val.detach().float()
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

                val_predictions.extend(predictions)
                val_labels.extend(references)
            
            train_predictions = []
            train_labels = []
            for step, batch in enumerate(final_train_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss_train = loss_fn(outputs.logits, batch['labels'].to('cuda'))
                    # We keep track of the loss at each epoch
                    if with_tracking:
                        total_loss_train += loss_train.detach().float()
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

                train_predictions.extend(predictions)
                train_labels.extend(references)

            eval_metric_acc = metric_acc.compute(predictions=val_predictions, references=val_labels)['accuracy']
            eval_metric_prec = metric_prec.compute(predictions=val_predictions, references=val_labels, average='macro')['precision']
            eval_metric_recall = metric_recall.compute(predictions=val_predictions, references=val_labels, average='macro')['recall']
            eval_metric_f1 = metric_f1.compute(predictions=val_predictions, references=val_labels, average='macro')['f1']

            train_metric_acc = metric_acc.compute(predictions=train_predictions, references=train_labels)['accuracy']
            train_metric_prec = metric_prec.compute(predictions=train_predictions, references=train_labels, average='macro')['precision']
            train_metric_recall = metric_recall.compute(predictions=train_predictions, references=train_labels, average='macro')['recall']
            train_metric_f1 = metric_f1.compute(predictions=train_predictions, references=train_labels, average='macro')['f1']

            logger.info(f"epoch {epoch} = Loss: {total_loss / len(final_train_dataloader)}, Loss_val: {total_loss_val / len(real_val_dataloader)}, Lr: {lr_scheduler.get_last_lr()[0]}, Accuracy: {eval_metric_acc}, Precision: {eval_metric_prec}, Recall: {eval_metric_recall}, F1: {eval_metric_f1}")

            if with_tracking:
                accelerator.log(
                    {
                        "accuracy": eval_metric_acc,
                        "precision": eval_metric_prec,
                        "recall": eval_metric_recall,
                        "f1": eval_metric_f1,
                        "train_loss": total_loss.item() / len(final_train_dataloader),
                        "val_loss": total_loss_val.item() / len(real_val_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                        "lr": lr_scheduler.get_last_lr()[0]
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
            train_loss_list.append(total_loss / len(final_train_dataloader))
            val_loss_list.append(total_loss_val / len(real_val_dataloader))

            if label_smoothing:
                loss_fn.next_epoch()

        # Save the final metrics
        train_acc.append(train_metric_acc)
        val_acc.append(eval_metric_acc)
        train_pre.append(train_metric_prec)
        val_pre.append(eval_metric_prec)
        train_rec.append(train_metric_recall)
        val_rec.append(eval_metric_recall)
        train_f1.append(train_metric_f1)
        val_f1.append(eval_metric_f1)
        
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

            all_results = {
                "accuracy": eval_metric_acc,
                "precision": eval_metric_prec,
                "recall": eval_metric_recall,
                "f1": eval_metric_f1,
            }
            with open(os.path.join(out_dir, "validation_results.json"), "w") as f:
                json.dump(all_results, f)

        # Save loss 
        train_loss_list_cpu = [x.cpu().numpy() if torch.is_tensor(x) else x for x in train_loss_list]
        val_loss_list_cpu = [x.cpu().numpy() if torch.is_tensor(x) else x for x in val_loss_list]

        plt.figure(figsize=(10, 6))
        plt.plot(list(range(1, num_of_epochs+1)), train_loss_list_cpu, 'bo-', label='Training loss')
        plt.plot(list(range(1, num_of_epochs+1)), val_loss_list_cpu, 'ro-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(out_dir + '/train_val_loss.png')


    names = []
    for xs in itertools.product(proportions_real, proportions_generated):
       if xs[0] == xs[1] == 0:
           continue
       names.append(f"Real{int(xs[0]*100)}-Generated{int(xs[1]*100)}") 

    
    plt.clf()
    # Plot for Accuracy
    plt.figure(figsize=(22, 16))
    plt.plot(names, train_acc, marker='o', linestyle='-', color='b', label='Train')
    plt.plot(names, val_acc, marker='s', linestyle='-', color='m', label='Validation')
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')  
    plt.tick_params(axis='x', which='major', labelsize='small')
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{config['exp_dir']}/logs/{config['name']}/accuracy_evolution.png")
    plt.clf()

    # Plot for Precision
    plt.figure(figsize=(22, 16))
    plt.plot(names, train_pre, marker='o', linestyle='-', color='b', label='Train')
    plt.plot(names, val_pre, marker='s', linestyle='-', color='m', label='Validation')
    plt.title('Precision', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')  
    plt.tick_params(axis='x', which='major', labelsize='small')
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{config['exp_dir']}/logs/{config['name']}/precision_evolution.png")
    plt.clf()

    # Plot for Recall
    plt.figure(figsize=(22, 16))
    plt.plot(names, train_rec, marker='o', linestyle='-', color='b', label='Train')
    plt.plot(names, val_rec, marker='s', linestyle='-', color='m', label='Validation')
    plt.title('Recall', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')  
    plt.tick_params(axis='x', which='major', labelsize='small')
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{config['exp_dir']}/logs/{config['name']}/recall_evolution.png")
    plt.clf()

    # Plot for F1 Score
    plt.figure(figsize=(22, 16))
    plt.plot(names, train_f1, marker='o', linestyle='-', color='b', label='Train')
    plt.plot(names, val_f1, marker='s', linestyle='-', color='m', label='Validation')
    plt.title('F1 Score', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')  
    plt.tick_params(axis='x', which='major', labelsize='small')
    plt.grid(True)
    plt.legend(['Train', 'Validation'])
    plt.savefig(f"{config['exp_dir']}/logs/{config['name']}/f1_evolution.png")
    plt.clf()

    # Create and save a csv file 
    df = pd.DataFrame({
        "Real-Generated": names,
        "Train Accuracy": train_acc,
        "Val Accuracy": val_acc,
        "Train Precision": train_pre,
        "Val Precision": val_pre,
        "Train Recall": train_rec,
        "Val Recall": val_rec,
        "Train F1": train_f1,
        "Val F1": val_f1
    })
    df.to_csv(f"{config['exp_dir']}/logs/{config['name']}/metrics.csv", index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)