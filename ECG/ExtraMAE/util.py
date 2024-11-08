import numpy as np
from einops import repeat  # , rearrange
import torch


def random_indexes(size):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0,
                        repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))
