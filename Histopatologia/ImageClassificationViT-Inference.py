"""
Inference and evauation of the ViT model 
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import datasets
from datasets import Dataset, Image, load_metric
#import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
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
from transformers import ViTImageProcessor, SwinForImageClassification

from diffusion.util import load_config 


DIRS = ['results']

logger = get_logger(__name__)

def main(config):

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{dir}", exist_ok=True)  
    out_dir = f"{BASE_DIR}/{dir}"   

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

    # Load pretrained model and image processor
    model = SwinForImageClassification.from_pretrained(f'{BASE_DIR}/model')
    model.to('cuda')
    image_processor = ViTImageProcessor.from_pretrained(f'{BASE_DIR}/model')

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

    if 'max_train_samples' in config["dataset"]:
        train_dataset = train_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_train_samples']))
    # Set the training transforms
    train_dataset.set_transform(train_transforms)
    if 'max_eval_samples' in config["dataset"]:
        val_dataset = val_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_eval_samples']))
    # Set the validation transforms
    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size']
    )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size'])
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=config['dataset']["dataloader"]['batch_size'])

    # Load metrics 
    metric_acc = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/accuracy.py")
    metric_prec = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/precision.py")
    metric_recall = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/recall.py")
    metric_f1 = load_metric("/gpfs/projects/bsc70/bsc70174/PANDA_code/diffusion/metrics/f1.py")


    #
    # Train Inference 
    #
    predictions_all = []
    references_all = []
    model.eval()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

        predictions_all.extend(predictions.tolist())
        references_all.extend(references.tolist())

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

    train_metric_acc = metric_acc.compute()
    train_metric_prec = metric_prec.compute(average='macro')
    train_metric_recall = metric_recall.compute(average='macro')
    train_metric_f1 = metric_f1.compute(average='macro')

    train_results = {
        "accuracy": list(train_metric_acc.values())[0],
        "precision": list(train_metric_prec.values())[0],
        "recall": list(train_metric_recall.values())[0],
        "f1": list(train_metric_f1.values())[0],
    }
    with open(os.path.join(out_dir, "train_results.json"), "w") as f:
        json.dump(train_results, f)

    # Generate the confusion matrix
    cm = confusion_matrix(references_all, predictions_all)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(f"{out_dir}/train_cm.png")


    #
    # Validation Inference 
    #
    predictions_all = []
    references_all = []
    model.eval()
    for step, batch in enumerate(tqdm(val_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

        predictions_all.extend(predictions.tolist())
        references_all.extend(references.tolist())

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

    val_metric_acc = metric_acc.compute()
    val_metric_prec = metric_prec.compute(average='macro')
    val_metric_recall = metric_recall.compute(average='macro')
    val_metric_f1 = metric_f1.compute(average='macro')

    val_results = {
        "accuracy": list(val_metric_acc.values())[0],
        "precision": list(val_metric_prec.values())[0],
        "recall": list(val_metric_recall.values())[0],
        "f1": list(val_metric_f1.values())[0],
    }
    with open(os.path.join(out_dir, "validation_results.json"), "w") as f:
        json.dump(val_results, f)

    # Generate the confusion matrix
    cm = confusion_matrix(references_all, predictions_all)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(f"{out_dir}/validation_cm.png")


    #
    # Test Inference 
    #
    predictions_all = []
    references_all = []
    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))

        predictions_all.extend(predictions.tolist())
        references_all.extend(references.tolist())

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

    test_metric_acc = metric_acc.compute()
    test_metric_prec = metric_prec.compute(average='macro')
    test_metric_recall = metric_recall.compute(average='macro')
    test_metric_f1 = metric_f1.compute(average='macro')

    test_results = {
        "accuracy": list(test_metric_acc.values())[0],
        "precision": list(test_metric_prec.values())[0],
        "recall": list(test_metric_recall.values())[0],
        "f1": list(test_metric_f1.values())[0],
    }
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f)

    # Generate the confusion matrix
    cm = confusion_matrix(references_all, predictions_all)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(f"{out_dir}/test_cm.png")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)