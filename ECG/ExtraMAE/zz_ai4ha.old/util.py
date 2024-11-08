import json
from time import time
import importlib
import numpy as np
import torch


def load_config(nfile):
    """
    Loads configuration file
    """

    ext = '.json' if 'json' not in nfile else ''

    f = open(nfile + ext, 'r')
    return json.load(f)


def time_management(last_time, qe_time, time_budget, logger):
    """
    Keeps the training time
    """
    epoch_time = (time() - last_time)/60.0
    last_time = time()
    if len(qe_time) > 10:
        qe_time.pop(0)
    qe_time.append(epoch_time)
    time_budget -= epoch_time
    hours = int(time_budget/60.0)
    mins = time_budget - (int(time_budget/60.0) * 60.0)
    logger.info(
        "--------------------- TIME MANAGEMENT --------------------------------------------------")
    logger.info(
        f'Remaining time budget: {hours:02d}h {mins:3.2f}m - mean iteration time {np.mean(qe_time):3.2f}m')
    logger.info(
        "--------------------- TIME MANAGEMENT --------------------------------------------------")
    return last_time, qe_time, time_budget


def print_memory(logger, accelerator, where):
    logger.info(f'MEM: memory allocated: {torch.cuda.memory_allocated(device=accelerator.device)/(1014*1024)}')
    logger.info(f'MEM: --------- {where} ------------------\n')
    logger.info(f'MEM: \n {torch.cuda.memory_summary(device=accelerator.device)}')
    logger.info('MEM: ---------------------------')  

# Functions taken from Latent Diffusion code
def instantiate_from_config(config):
    if not "class" in config:
        raise KeyError("Expected key `class` to instantiate.")
    return get_obj_from_str(config["class"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def dataset_first_batch_std(data, config, key):
    config['shuffle'] = False
    config['batch_size'] *= 8
    dataloader = torch.utils.data.DataLoader(data, **config)
    for batch in dataloader:
        images = batch[key]
        break

    return images.std()
   

def dataset_first_batch_dl_std(dataloader, key):
    for batch in dataloader:
        images = batch[key]
        break

    return images.std()

def dataset_statistics(dataloader, key, channels=3, cuda=True):
    """_Computes mean and std of dataset using a dataloader_

    Args:
        dataloader (_type_): _description_
    """
    dim = list(range(channels))
    dim.remove(1)

    cnt = 0
    if cuda:
        fst_moment = torch.empty(channels).to('cuda')
        snd_moment = torch.empty(channels).to('cuda')
    else:
        fst_moment = torch.empty(channels)
        snd_moment = torch.empty(channels)


    for batch in dataloader:
        images = batch[key]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=dim)
        sum_of_square = torch.sum(images ** 2,
                                  dim=dim)
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std


if __name__ == '__main__':
    load_config("/home/bejar/PycharmProjects/misiones/Imagenes/Training/jobs/DFLDPolyp-seg-DDIM-s128-l64-123-r3-linear-t1000.json")