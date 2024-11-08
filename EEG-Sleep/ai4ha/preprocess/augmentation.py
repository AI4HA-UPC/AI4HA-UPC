# This file contains the augmentation functions for the AI4HA project
from collections import Counter


def augmentation_mix_up(batch, labels, coef):
    """_Computes mix up augmentation of a batch of examples_
    
     Simple mix-up augmentation that pairs examples, and if they are 
     from different class augments the less frequent as the convex
     combination of both with the coeff

    Args:
        batch (_type_): _batch of data_
        labels (_type_): _labels of the data_
        coef (_type_): _mixing coefficient_
    """
    count = Counter(list(labels.cpu().numpy()))
    for i in range(0, batch.shape[0], 2):
        if labels[i] != labels[i + 1]:
            sb1, sb2 = (i, i +
                        1) if count[labels[i]] > count[labels[i +
                                                              1]] else (i + 1,
                                                                        i)

            batch[sb2] = (batch[sb1] * (1.0 - coef)) + (batch[sb2] * coef)
    return batch
