import torch
import os
import numpy as np
import json
import yaml

def load_config(nfile):
    """
    Loads configuration file
    """
    nfile = nfile + '.yaml'
    with open(nfile, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return exc

def local_directory(name=None, diffusion_cfg=None, output_directory=None, T=0, beta_0=0, beta_T=0):

    #TODO: Change names and params
    model_name = name
    local_path = f'{model_name}_T_{T}_beta_0_{beta_0}_beta_T_{beta_T}'

    # Get shared output_directory ready
    output_directory = os.path.join("experiments/", local_path, output_directory)
    os.makedirs(output_directory, mode=0o775, exist_ok=True)
    print("output directory", output_directory, flush=True)
    return local_path, output_directory


def print_size(net, verbose=False):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        module_parameters = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        if verbose:
            for n, p in module_parameters:
                print(n, p.numel())

        params = sum([np.prod(p.size()) for n, p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)