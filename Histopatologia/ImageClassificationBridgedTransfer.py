import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from PIL import Image
import itertools
import random
import json

from sklearn.metrics import confusion_matrix
import torch 
from datasets import load_metric, DatasetDict, Dataset, Image, concatenate_datasets
from torch.utils.data import random_split
from transformers import ViTImageProcessor, EfficientNetForImageClassification, AutoConfig
from torchvision import transforms
import evaluate
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion.normalization.reinhard import NormalizerTransform
from diffusion.losses.weighted_focal_loss import FocalLoss

from tqdm import tqdm 


STEPS = ['step1', 'step2']
DIRS = ['checkpoints','logs', "model", "best_model"]

MODELS = {
    'swinv2': SwinForImageClassification,
    'resnet': ResNetForImageClassification,
    'convnext': ConvNextForImageClassification,
    'efficientnet': EfficientNetForImageClassification,
    'convnext2': ConvNextV2ForImageClassification,
    #'pvtv2': PvtV2ForImageClassification
}

logger = get_logger(__name__)


def main(config):

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"

    for step in STEPS:
        os.makedirs(f"{BASE_DIR}/{step}", exist_ok=True)  
    for dir in DIRS:
        os.makedirs(f"{BASE_DIR}/{step}/{dir}", exist_ok=True)  

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

    
    label = config["model"]["label"] 
    # Prepare label mappings.
    id2label = config["dataset"]["id2label"]
    label2id = {v: k for k, v in id2label.items()} 

    image_processor = ViTImageProcessor.from_pretrained(config['model']['model_path'])

    # Load pretrained model and image processor
    config_model = AutoConfig.from_pretrained(config['model']['model_path'])
    config_model.id2label = id2label
    config_model.label2id = label2id

    #   
    # Step 1
    #
    out_dir = f"{BASE_DIR}/{STEPS[0]}" 

    gen_train_df = pd.read_csv(config['gen_dataset']['train']['csv_path'])
    gen_test_df = pd.read_csv(config['gen_dataset']['test']['csv_path'])

    gen_train_image_list = gen_train_df['Image_id'].tolist()
    gen_train_image_list = [os.path.join(config['gen_dataset']['train']['data_path'], 'patches/'+img)+'.jpg' for img in gen_train_image_list]

    gen_test_image_list = gen_test_df['Image_id'].tolist()
    gen_test_image_list = [os.path.join(config['gen_dataset']['test']['data_path'], 'patches/'+img)+'.jpg' for img in gen_test_image_list]

    gen_train_label_list = gen_train_df[label].tolist()
    gen_test_label_list = gen_test_df[label].tolist()

    gen_train_dataset = Dataset.from_dict({"image": gen_train_image_list, "label": gen_train_label_list}).cast_column("image", datasets.Image())
    gen_test_dataset = Dataset.from_dict({"image": gen_test_image_list, "label": gen_test_label_list}).cast_column("image", datasets.Image())

    # If we don't have a validation split, split off a percentage of train as validation.
    if 'train_val_split' in config['gen_dataset']:
        if isinstance(config['gen_dataset']['train_val_split'], float) and config['gen_dataset']['train_val_split'] > 0.0:
            splits = gen_train_dataset.train_test_split(test_size=config['gen_dataset']['train_val_split'])
            gen_train_dataset = splits['train']
            gen_val_dataset = splits['test']

    # Load pre-trained model 
    logger.info("Loading model...")
    model = MODELS[config['model']['model_name']].from_pretrained(
        config['model']['model_path'],
        num_labels=config['model']['num_classes'],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    image_mean, image_std = image_processor.image_mean, image_processor.image_std
    if config['model']['model_name'] == 'convnext2':
        size = image_processor.size["shortest_edge"]
    else:
        size = image_processor.size["height"]

    # CutMix 
    if config['train']['mixup']:
        cutmix = v2.CutMix(num_classes=config['model']['num_classes'])
    
    stain_normalization = NormalizerTransform()
    stain_normalization.fit(image_path='/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/normalization/reference_img.png')
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

    if 'max_train_samples' in config["gen_dataset"]:
        gen_train_dataset = gen_train_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_train_samples']))
    # Set the training transforms
    gen_train_dataset.set_transform(train_transforms)

    if 'max_eval_samples' in config["gen_dataset"]:
        gen_val_dataset = gen_val_dataset.shuffle(seed=config['seed']).select(range(config["dataset"]['max_eval_samples']))
    # Set the validation transforms
    gen_val_dataset.set_transform(val_transforms)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    if config['train']['mixup']:
        def collate_fn_cutmix(examples):
            return cutmix(collate_fn(examples))

        train_dataloader = DataLoader(
            gen_train_dataset, shuffle=True, collate_fn=collate_fn_cutmix, batch_size=config['gen_dataset']["dataloader"]['batch_size']
        )
        #val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_cutmix, batch_size=config['dataset']["dataloader"]['batch_size'])

    else:
        train_dataloader = DataLoader(
            gen_train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config['gen_dataset']["dataloader"]['batch_size']
        )
    val_dataloader = DataLoader(gen_val_dataset, collate_fn=collate_fn, batch_size=config['gen_dataset']["dataloader"]['batch_size'])

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

    # Label Smoothing 
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
                num_classes=config['model']['num_classes'],
                mode='quadratic'
            )
        elif config['train']['loss'] == 'focal':
            loss_fn = FocalLoss()

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
    metric_acc = evaluate.load("/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/metrics/accuracy.py", experiment_id=config['name'])
    metric_prec = evaluate.load("/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/metrics/precision.py", experiment_id=config['name'])
    metric_recall = evaluate.load("/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/metrics/recall.py", experiment_id=config['name'])
    metric_f1 = evaluate.load("/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/metrics/f1.py", experiment_id=config['name'])

    # Train!
    total_batch_size = config['gen_dataset']["dataloader"]["batch_size"] * accelerator.num_processes * accparams["gradient_accumulation_steps"]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['train']['epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['gen_dataset']['dataloader']['batch_size']}")
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

    hps = {"num_iterations": config['train']['epochs'], "learning_rate": config['train']['learning_rate'] * accelerator.num_processes}
    accelerator.init_trackers(config['name'], config=hps, 
                              init_kwargs={"wandb":
                                           {"dir":os.path.join(BASE_DIR, "logs")}})
    with_tracking = True   

    checkpoints_dir = f"{BASE_DIR}/checkpoints"