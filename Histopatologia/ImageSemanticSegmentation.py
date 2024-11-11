"""
Script for the training from scratch of semantic segmentation models 
""" 
import argparse
import json
import logging
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datasets
from datasets import Dataset, Image, load_metric
import torch
from torch import optim, nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import albumentations as A
import evaluate

import transformers
from transformers import AutoConfig, SegformerImageProcessor, SchedulerType, get_scheduler, SegformerForSemanticSegmentation
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch.nn as nn

from diffusion.util import load_config


DIRS = ['checkpoints','logs', "model", "samples"]

MODELS = {
    'segformer': SegformerForSemanticSegmentation
}

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

    train_image_names_list = train_df['Image_id'].tolist()
    train_image_list = [os.path.join(config['dataset']['train']['data_path'], 'patches/'+img)+'.png' for img in train_image_names_list]

    test_image_names_list = test_df['Image_id'].tolist()
    test_image_list = [os.path.join(config['dataset']['test']['data_path'], 'patches/'+img)+'.png' for img in test_image_names_list]

    train_mask_list = [os.path.join(config['dataset']['train']['data_path'], 'masks/'+img)+'_mask.png' for img in train_image_names_list]
    test_mask_list = [os.path.join(config['dataset']['test']['data_path'], 'masks/'+img)+'_mask.png' for img in test_image_names_list]

    train_dataset = Dataset.from_dict({"image": train_image_list, "mask": train_mask_list}).cast_column("image", Image()).cast_column("mask", Image())
    test_dataset = Dataset.from_dict({"image": test_image_list, "mask": test_mask_list}).cast_column("image", Image()).cast_column("mask", Image())

    # If we don't have a validation split, split off a percentage of train as validation.
    if 'train_val_split' in config['dataset']:
        if isinstance(config['dataset']['train_val_split'], float) and config['dataset']['train_val_split'] > 0.0:
            splits = train_dataset.train_test_split(test_size=config['dataset']['train_val_split'])
            train_dataset = splits['train']
            val_dataset = splits['test']

    # Prepare label mappings.
    id2label = config["dataset"]["id2label"]
    label2id = {v: k for k, v in id2label.items()} 

    image_processor = SegformerImageProcessor.from_pretrained(config['model']['model_path'])
    image_processor.do_resize = True
    image_processor.size['height'] = 256
    image_processor.size['width'] = 256

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
    
    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    img_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Downscale(p=0.1, scale_min=0.4, scale_max=0.6),
                    A.GaussNoise(p=0.2),
                ],
                p=0.1,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.2),
                    A.ColorJitter(p=0.2),
                    A.HueSaturationValue(p=0.2),
                ],
                p=0.1,
            ),
            A.OneOf([A.PixelDropout(p=0.2), A.RandomGravel(p=0.2)], p=0.15),
        ]
    )

    def train_transforms(example_batch):
        """
        Transform the dataset for training.

        Args:
            example_batch: Batch of examples from the dataset.

        Returns:
            A dictionary of the inputs to the model.
        """
        trans = [
            img_transforms(image=np.array(x), mask=np.array(m))
            for x, m in zip(example_batch["image"], example_batch["mask"])
        ]
        images = [x["image"] for x in trans]
        labels = [x["mask"] for x in trans]
        inputs = image_processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        """
        Transform the dataset for validation.

        Args:
            example_batch: Batch of examples from the dataset.

        Returns:
            A dictionary of the inputs to the model.
        """
        images = [x for x in example_batch["image"]]
        labels = [x for x in example_batch["mask"]]
        inputs = image_processor(images, labels)
        return inputs

    if 'max_train_samples' in config["dataset"]:
        train_dataset = train_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_train_samples']))
    # Set the training transforms
    train_dataset.set_transform(train_transforms)

    if 'max_eval_samples' in config["dataset"]:
        val_dataset = val_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_eval_samples']))
    # Set the validation transforms
    val_dataset.set_transform(val_transforms)

    def inverse_segformer_process(image):
        image_mean = np.array(image_processor.image_mean).reshape(3, 1, 1)
        image_std = np.array(image_processor.image_std).reshape(3, 1, 1)

        # Reverse Rescaling
        #image /= image_processor.rescale_factor  # Perform element-wise division with rescale

        # Reverse Normalization
        image *= image_std  # Perform element-wise multiplication with image_std
        image += image_mean  # Perform element-wise addition with image_mean

        return image

    # DataLoaders creation:
    #def collate_fn(examples):
    #    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #    labels = torch.tensor([example["label"] for example in examples])
    #    return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config['dataset']["dataloader"]['batch_size']
    )
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config['dataset']["dataloader"]['batch_size'])

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])

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

    # Loss function 
    loss_func = nn.CrossEntropyLoss()

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

    # Get the metrics
    #metric_miou = load_metric("/media/axelrom16/Axel/Treball/bsc70174/PANDA_code/diffusion/metrics/mean_iou.py", experiment_id=config['name'])
    metric_miou = evaluate.load("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/mean_iou.py")

    # Train!
    total_batch_size = config['dataset']["dataloader"]["batch_size"] * accelerator.num_processes * accparams["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['train']['epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['dataset']['dataloader']['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    print()
    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype)

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
                labels = batch["labels"].to('cuda')
                # Increase the size of the mask
                logits_tensor = nn.functional.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = loss_func(logits_tensor, labels)
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.item()
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

        model.eval()
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                labels = batch["labels"].to('cuda')
                logits_tensor = nn.functional.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss_val = loss_func(logits_tensor, labels)
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss_val += loss_val.item()
            
            predictions = logits_tensor.argmax(dim=1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

            metric_miou.add_batch(
                predictions=predictions,
                references=references,
            )

            if step == 0:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(inverse_segformer_process(batch['pixel_values'][0].cpu()).numpy().transpose(1, 2, 0))
                axs[0].set_title("Image")
                axs[0].axis("off")
                axs[1].imshow(labels[0].unsqueeze(0).cpu().numpy().transpose(1,2,0), cmap="gray", vmin=0, vmax=5)
                axs[1].set_title("Label")
                axs[1].axis("off")
                axs[2].imshow(predictions[0].unsqueeze(0).cpu().numpy().transpose(1,2,0), cmap="gray", vmin=0, vmax=5)
                axs[2].set_title("Prediction")
                axs[2].axis("off")
                plt.tight_layout()
                plt.savefig(BASE_DIR + f'/samples/sample_{epoch}.png')

        eval_metric_miou = metric_miou.compute(num_labels=config['model']['num_classes'], ignore_index=255)
        logger.info(f"epoch {epoch} = Loss: {total_loss / len(train_dataloader)}, Lr: {lr_scheduler.get_last_lr()[0]}, Mean Iou: {list(eval_metric_miou.values())[0]}")

        if with_tracking:
            accelerator.log(
                {
                    "miou": list(eval_metric_miou.values())[0],
                    "train_loss": total_loss / len(train_dataloader),
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
        train_loss_list.append(total_loss / len(train_dataloader))
        val_loss_list.append(total_loss_val / len(val_dataloader))

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
            "miou": list(eval_metric_miou.values())[0]
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