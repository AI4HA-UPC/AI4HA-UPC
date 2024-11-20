from typing import Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(ResidualBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv1d(self.num_neurons, self.num_neurons, (5,), padding="same")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class ConvolutionalBlock(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(ConvolutionalBlock, self).__init__()
        self.residual_block = ResidualBlock(num_neurons)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(5, stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class ResnetSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        data_t_steps: int,
        num_conv_blocks: int = 5,
        num_neurons: int = 32,
        num_classes: int = 5,
    ):
        super(ResnetSignalClassification, self).__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_neurons = num_neurons
        for i in range(num_conv_blocks):
            data_t_steps = int((data_t_steps - 5) / 2) + 1

        self.input = nn.Conv1d(1, self.num_neurons, (5,), padding="same")
        self.conv_blocks = nn.ModuleList(
            [ConvolutionalBlock(self.num_neurons) for _ in range(num_conv_blocks)]
        )
        self.linear1 = nn.Linear(self.num_neurons * data_t_steps, self.num_neurons)
        self.linear2 = nn.Linear(self.num_neurons, self.num_neurons // 2)
        self.linear3 = nn.Linear(self.num_neurons // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(inplace=True, p=0.5)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, time = x.shape
        x = x.view(bsz, -1)
        x = self.linear1(x)
        # self.dropout_1(x)
        self.relu(x)
        x = self.linear2(x)
        # self.dropout_1(x)
        self.relu(x)
        x = self.linear3(x)
        y = self.output(x)
        return y

    def freeze_all_but_last(self):
        # named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if "output" in n or "linear3" in n:
                pass
            else:
                p.requires_grad = False


class NBEATSBlock(nn.Module):
    def __init__(self, num_neurons: int, horizon: int, seq_t_steps: int):
        super(NBEATSBlock, self).__init__()
        self.num_neurons = num_neurons

        self.linear = nn.Linear(self.num_neurons, self.num_neurons)
        self.linear_back = nn.Linear(self.num_neurons, seq_t_steps)
        self.linear_forecast = nn.Linear(self.num_neurons, horizon)

    def forward(self, x):
        # FC stack
        x = self.linear(x)
        self.relu(x)
        x = self.linear(x)
        self.relu(x)
        x = self.linear(x)
        self.relu(x)
        x = self.linear(x)
        self.relu(x)

        backcast_params = self.linear(x)
        forecast_params = self.linear(x)

        # generic architecture
        x_hat = self.linear_back(backcast_params)
        y_hat = self.linear_forecast(forecast_params)

        return x_hat, y_hat


class NBEATSStack(nn.Module):
    def __init__(self, num_neurons: int, horizon: int, seq_t_steps: int, num_blocks):
        super(NBEATSStack, self).__init__()
        self.num_neurons = num_neurons
        self.horizon = horizon

        self.nbeats_blocks = nn.ModuleList(
            [NBEATSBlock(num_neurons, horizon, seq_t_steps) for _ in range(num_blocks)]
        )

    def forward(self, x):
        stack_y_hat = torch.zeros(1, 1, self.horizon)
        for n, block in enumerate(self.nbeats_blocks):
            x_hat, y_hat = block(x)
            x = x - x_hat
            stack_y_hat = stack_y_hat + y_hat
        return x, stack_y_hat


class NBEATS(nn.Module):
    """
    Implementation of the model proposed in https://arxiv.org/pdf/1905.10437.pdf
    """

    def __init__(
        self,
        data_t_steps: int,
        num_classes: int,
        num_neurons: int = 32,
        num_stacks: int = 2,
        num_blocks: int = 4,
        horizon: int = 7,
        seq_t_steps: int = 14,
    ):
        super(NBEATS, self).__init__()

        self.nbeats_stacks = nn.ModuleList(
            [
                NBEATSStack(num_neurons, horizon, seq_t_steps, num_blocks)
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x):
        final_y_hat = torch.zeros(1, 1, self.horizon)
        for n, stack in enumerate(self.nbeats_stacks):
            x, y_hat = stack(x)
            final_y_hat = final_y_hat + y_hat
        return final_y_hat


class Residual2DBlock_1x1_convolution(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(Residual2DBlock_1x1_convolution, self).__init__()

        self.num_neurons = num_neurons
        self.conv1_in = nn.Conv2d(
            self.num_neurons, self.num_neurons, (1, 1), padding="same"
        )
        self.conv3 = nn.Conv2d(
            self.num_neurons, self.num_neurons, (3, 3), padding="same"
        )
        self.conv1_out = nn.Conv2d(
            self.num_neurons, self.num_neurons * 2, (1, 1), padding="same"
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv1_in(x)
        self.relu(x)
        x = self.conv3(x)
        self.relu(x)
        x = x + original_x
        x = self.conv1_out(x)
        return x


class Convolutional2DBlock_1x1_convolution(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(Convolutional2DBlock_1x1_convolution, self).__init__()
        self.residual_block = Residual2DBlock_1x1_convolution(num_neurons)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((2, 5), stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class Resnet2DSignalClassification_1x1_convolution(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        num_neurons: int = 32,
        num_classes: int = 5,
        dropout_rate: float = 0,
    ):
        super(Resnet2DSignalClassification_1x1_convolution, self).__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_neurons = num_neurons
        for i in range(num_conv_blocks):
            data_d_steps = int((data_d_steps - 2) / 2) + 1
            data_h_steps = int((data_h_steps - 2) / 2) + 1
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_d_steps * data_h_steps * data_t_steps

        self.input_together = nn.Conv2d(1, self.num_neurons, (3, 3), padding="same")
        self.input_individual = nn.Conv1d(1, self.num_neurons, (5,), padding="same")
        self.input_max_pool = nn.MaxPool2d((3, 3), stride=2)
        self.conv_blocks = nn.ModuleList(
            [
                Convolutional2DBlock_1x1_convolution(self.num_neurons * 2**i)
                for i in range(num_conv_blocks)
            ]
        )
        self.linear1 = nn.Linear(self.num_neurons * 2**num_conv_blocks, num_classes)
        # self.linear2 = nn.Linear(self.num_neurons, self.num_neurons // 2)
        # self.linear3 = nn.Linear(self.num_neurons // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(inplace=True, p=dropout_rate)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        splits = x.chunk(12, 2)
        group = []
        for split in splits:
            split = split.squeeze(2)
            self.input_individual(split)
            self.relu(x)
            split = split.unsqueeze(2)
            group.append(split)

        x = torch.concatenate(group, 2)
        x = self.input_together(x)
        self.relu(x)
        # x = self.input_max_pool(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, h, time = x.shape
        x = x.mean([2, 3])  # GAP
        x = self.linear1(x)
        y = self.output(x)
        return y

    def freeze_all_but_last(self):
        # named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if "output" in n or "linear3" in n:
                pass
            else:
                p.requires_grad = False


class Residual3DBlock(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(Residual3DBlock, self).__init__()

        self.num_neurons = num_neurons
        self.conv = nn.Conv3d(
            self.num_neurons, self.num_neurons, (5, 5, 5), padding="same"
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        self.relu(x)
        x = self.conv(x)
        return x + original_x


class Convolutional3DBlock(nn.Module):
    def __init__(self, num_neurons: int = 32):
        super(Convolutional3DBlock, self).__init__()
        self.residual_block = Residual3DBlock(num_neurons)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d((2, 1, 5), stride=2)

    def forward(self, x):
        x = self.residual_block(x)
        self.relu(x)
        x = self.max_pool(x)
        return x


class Resnet3DSignalClassification(nn.Module):
    """
    Implementation of the model presented in https://arxiv.org/pdf/1805.00794.pdf
    """

    def __init__(
        self,
        data_d_steps: int,
        data_h_steps: int,
        data_t_steps: int,
        num_conv_blocks: int = 1,
        num_neurons: int = 32,
        num_classes: int = 5,
    ):
        super(Resnet3DSignalClassification, self).__init__()

        self.num_conv_blocks = num_conv_blocks
        self.num_neurons = num_neurons
        for i in range(num_conv_blocks):
            data_d_steps = int((data_d_steps - 2) / 2) + 1
            data_h_steps = int((data_h_steps - 1) / 2) + 1
            data_t_steps = int((data_t_steps - 5) / 2) + 1
            data_total_steps = data_d_steps * data_h_steps * data_t_steps

        self.input = nn.Conv3d(1, self.num_neurons, (5, 5, 5), padding="same")
        self.conv_blocks = nn.ModuleList(
            [Convolutional3DBlock(self.num_neurons) for _ in range(num_conv_blocks)]
        )
        self.linear1 = nn.Linear(self.num_neurons * data_total_steps, self.num_neurons)
        self.linear2 = nn.Linear(self.num_neurons, self.num_neurons // 2)
        self.linear3 = nn.Linear(self.num_neurons // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(inplace=True, p=0.5)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, depth, h, time = x.shape
        x = x.view(bsz, -1)
        x = self.linear1(x)
        # self.dropout_1(x)
        self.relu(x)
        x = self.linear2(x)
        # self.dropout_1(x)
        self.relu(x)
        x = self.linear3(x)
        y = self.output(x)
        return y

    def freeze_all_but_last(self):
        # named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if "output" in n or "linear3" in n:
                pass
            else:
                p.requires_grad = False


if __name__ == "__main__":
    univariate_model = False
    test_resnet = False
    test_2dresnet = True
    test_3dresnet = False
    test_nbeats = False
    width_sequence = 1000
    if univariate_model:
        if test_resnet:
            num_cnn_blocks = 8
            m = ResnetSignalClassification(
                width_sequence, num_conv_blocks=num_cnn_blocks
            )
            t = torch.rand((2, 1, width_sequence))
            tes = m(t)
            print(tes.shape)
        elif test_nbeats:
            m = NBEATS()
    else:
        if test_2dresnet:
            num_cnn_blocks = 3
            depth_sequence = 1
            height_sequence = 12
            m = Resnet2DSignalClassification(
                depth_sequence,
                height_sequence,
                width_sequence,
                num_conv_blocks=num_cnn_blocks,
            )
            t = torch.rand((2, 1, depth_sequence, height_sequence, width_sequence))
            tes = m(t.squeeze().unsqueeze(axis=1))
            print(tes.shape)

        if test_3dresnet:
            num_cnn_blocks = 10
            depth_sequence = 12
            height_sequence = 1
            m = Resnet3DSignalClassification(
                depth_sequence,
                height_sequence,
                width_sequence,
                num_conv_blocks=num_cnn_blocks,
            )
            t = torch.rand((2, 1, depth_sequence, height_sequence, width_sequence))
            tes = m(t)
            print(tes.shape)
