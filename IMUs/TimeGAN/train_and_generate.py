import itertools
import argparse

from tqdm import tqdm
from functools import partial
from trainings.timegan_trainingPrevStep import TimeGanTrainer
import torch
from itertools import combinations
from utils.utils import *  


def get_model(model_cfg):
    print(model_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if model_cfg['name'] == "timegan":
        #T = [20, 40, 50, 200]
        #beta_0 = [0.0001]
        #learning_rate = [0.1, 0.001, 0.0001, 0.00001]
        T = [20,]
        beta_0 = [0.0001]
        learning_rate = [ 0.0001]
        opt = list(itertools.product(T, beta_0, learning_rate))

        return TimeGanTrainer(device=device, model_cfg=model_cfg), opt

def train_config(rank, num_gpus, cfg):

    model_train, opt = get_model(cfg['model'])
    for option in opt:
        if cfg['model']['name'] == "timegan":
            if option[0] <= 50:  # T
                option = option + (0.05,)  # beta_T
            else:
                option = option + (0.02,)  # beta_T
            model = model_train.train_model(rank=rank, num_gpus=num_gpus, model_cfg=cfg['model'],
                                    diffusion_cfg=cfg['diffusion'], dataset_cfg=cfg['dataset'], train_cfg=cfg['train'],opt=option, generate_cfg=cfg['generate'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/config_hd4_nl3_100ep_crossPrevStep_test10", 
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config) 
    rank = 0
    num_gpus = 1
    train_config(rank, num_gpus, config) 
