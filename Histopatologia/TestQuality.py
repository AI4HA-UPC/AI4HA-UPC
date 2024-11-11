import argparse
import inspect
import logging
import math
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import torch
from diffusion.validation.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
from diffusion.util import instantiate_from_config, load_config


def main(config):
    accparams = config['accelerator']
    accelerator = Accelerator(**accparams)

    fid = FrechetInceptionDistance(feature=config["fid"]['n_features'],
                                   feature_extractor_weights_path=config['fid']['weights_path']).to('cuda')

    real_data = instantiate_from_config(config['dataset']['real'])
    generated_data = instantiate_from_config(config['dataset']['generated'])

    # Reduce the true dataset
    desired_dataset_size = len(real_data) // 3  
    random_indices = torch.randperm(len(real_data))[:desired_dataset_size]
    reduced_dataset = torch.utils.data.Subset(real_data, random_indices)

    print("Real dataset size: ", len(reduced_dataset))
    print("Generated dataset size: ", len(generated_data))

    real_dataloader = torch.utils.data.DataLoader(reduced_dataset, **config['dataset']["dataloader"])  
    dloader = accelerator.prepare(real_dataloader)

    for batch in tqdm(dloader):
        batch_img = batch['image']
        batch_img = ((batch_img / 2) + 0.5) * 255
        fid.update(batch_img.permute(0, 3, 1, 2).type(torch.uint8), real=True)

    generated_dataloader = torch.utils.data.DataLoader(generated_data, **config['dataset']["dataloader"])
    dloader = accelerator.prepare(generated_dataloader)
    for batch in tqdm(dloader):
        fid.update(batch[0].type(torch.uint8), real=False)

    print('FID=',fid.compute())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)