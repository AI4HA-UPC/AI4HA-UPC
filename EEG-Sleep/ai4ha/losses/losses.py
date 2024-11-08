import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Implementation of the Focal loss function

        Args:
            weight: class weight vector to be used in case of class imbalance
            gamma: hyper-parameter for the focal loss scaling.
    """

    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs,
                                  targets,
                                  reduction='none',
                                  weight=self.weight)

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt)**self.gamma *
                      ce_loss).mean()  # mean over the batch
        return focal_loss


class OrdinalLoss(nn.Module):
    """Implementation of the Ordinal loss function

        Args:
            nclasses: number of classes
    """
    def __init__(self, nclasses):
        super(OrdinalLoss, self).__init__()
        self.nclasses = nclasses

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs,
                                  targets,
                                  reduction='none')
        am_outs = torch.argmax(outputs, dim=1)
        am_outs = (torch.abs(am_outs - targets)/self.nclasses) + 1 # ordinal loss
        return (am_outs * ce_loss).mean()
