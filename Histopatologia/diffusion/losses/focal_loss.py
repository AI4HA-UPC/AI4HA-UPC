import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initializes the Weighted Focal Loss module.
        :param weight: A 1D tensor of shape `[C]` where C is the number of classes.
        :param alpha: Balancing factor, typically between 0 and 1.
        :param gamma: Focusing parameter, typically greater than 0.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the focal loss.
        :param inputs: Predictions from the model (before softmax).
        :param targets: Ground truth labels.
        """
        if self.weight is not None:
            if self.weight.type() != inputs.data.type():
                self.weight = self.weight.type_as(inputs.data)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
