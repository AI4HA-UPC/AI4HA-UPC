from typing import Tuple

import torch
from torch import nn


class Residual2DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: Tuple[int, int]):
        super(Residual2DBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv2d(
            self.num_neurons, self.num_neurons, filter_size, padding="same", dilation=5
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class Convolutional2DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: Tuple[int, int]):
        super(Convolutional2DBlock, self).__init__()
        self.residual_block = Residual2DBlock(num_neurons, filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((2, 5), stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class Resnet2DSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        num_kernels: int = 8,
        num_neurons: int = 32,
        num_classes: int = 5,
        filter_size: Tuple[int, int] = (5, 5),
        dropout_rate: float = 0.25,
    ):
        super(Resnet2DSignalClassification, self).__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_neurons = num_neurons
        self.num_kernels = num_kernels
        for i in range(num_conv_blocks):
            data_d_steps = int((data_d_steps - 2) / 2) + 1
            data_h_steps = int((data_h_steps - 2) / 2) + 1
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_d_steps * data_h_steps * data_t_steps

        self.input = nn.Conv2d(1, self.num_kernels, filter_size, padding="same")
        self.conv_blocks = nn.ModuleList(
            [
                Convolutional2DBlock(self.num_kernels, filter_size)
                for _ in range(num_conv_blocks)
            ]
        )
        self.linear1 = nn.Linear(self.num_kernels * data_total_steps, self.num_neurons)
        # self.linear2 = nn.Linear(self.num_neurons, self.num_neurons // 2)
        self.linear3 = nn.Linear(self.num_neurons, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(inplace=True, p=dropout_rate)
        # self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, h, time = x.shape
        x = x.view(bsz, -1)
        x = self.linear1(x)
        self.dropout_1(x)
        self.relu(x)
        # x = self.linear2(x)
        # self.dropout_1(x)
        # self.relu(x)
        logit = self.linear3(x)
        return logit


if __name__ == "__main__":
    univariate_model = False
    test_resnet = False
    test_2dresnet = True
    test_3dresnet = False
    test_nbeats = False
    width_sequence = 1000
    num_cnn_blocks = 3
    depth_sequence = 1
    height_sequence = 12
    m = Resnet2DSignalClassification(
        depth_sequence, height_sequence, width_sequence, num_conv_blocks=num_cnn_blocks
    )
    t = torch.rand((2, 1, depth_sequence, height_sequence, width_sequence))
    tes = m(t.squeeze().unsqueeze(axis=1))
    print(tes.shape)
