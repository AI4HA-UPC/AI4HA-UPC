from typing import Tuple

import torch
from torch import nn


class Residual1DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: int):
        super(Residual1DBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv1d(
            self.num_neurons, self.num_neurons, filter_size, padding="same", dilation=5
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class Convolutional1DBlock(nn.Module):
    def __init__(self, num_neurons: int, filter_size: int):
        super(Convolutional1DBlock, self).__init__()
        self.residual_block = Residual1DBlock(num_neurons, filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(5, stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class Resnet1DSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        n_kernels: int = 8,
        n_neurons_classifier: int = 32,
        num_layers_classifier: int = 3,
        num_classes: int = 5,
        filter_size: int = 5,
        dropout_rate: float = 0.25,
    ):
        super(Resnet1DSignalClassification, self).__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_neurons = n_neurons_classifier
        self.num_kernels = n_kernels
        self.data_d_steps = data_d_steps
        self.data_h_steps = data_h_steps
        self.data_t_steps = data_t_steps
        self.filter_size = filter_size
        self.num_layers_classifier = num_layers_classifier
        for i in range(num_conv_blocks):
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_t_steps

        self.input = nn.Conv1d(
            self.data_d_steps, self.num_kernels, self.filter_size, padding="same"
        )
        self.conv_blocks = nn.ModuleList(
            [
                Convolutional1DBlock(self.num_kernels, self.filter_size)
                for _ in range(num_conv_blocks)
            ]
        )

        in_features = self.num_kernels * data_total_steps
        layers = []
        for num_layers in range(self.num_layers_classifier):
            layers.append(nn.Linear(in_features, self.num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            in_features = self.num_neurons

        layers.append(nn.Linear(in_features, num_classes))
        self.classifier_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, time = x.shape
        x = x.view(bsz, -1)
        logit = self.classifier_layers(x)
        return logit

    def freeze_all_but_last(self):
        # named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if "classifier_layers" in n:
                pass
            else:
                p.requires_grad = False


if __name__ == "__main__":
    width_sequence = 1000
    num_cnn_blocks = 7
    num_neurons = 512
    num_kernels = 16
    classifier_neurons = 256
    depth_sequence = 12
    height_sequence = 1
    m = Resnet1DSignalClassification(
        depth_sequence,
        height_sequence,
        width_sequence,
        num_conv_blocks=num_cnn_blocks,
        n_kernels=num_kernels,
        n_neurons_classifier=num_neurons,
    )
    t = torch.rand((2, 1, depth_sequence, width_sequence, height_sequence))
    tes = m(t.squeeze())
    print(tes.shape)
