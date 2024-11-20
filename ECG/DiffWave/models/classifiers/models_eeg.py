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


class ResnetSignalClassificationEEG(nn.Module):
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
        super(ResnetSignalClassificationEEG, self).__init__()

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
        self.dropout_1 = nn.Dropout(inplace=True, p=0.2)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.input(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        bsz, nch, time = x.shape
        x = x.view(bsz, -1)
        x = self.linear1(x)
        self.dropout_1(x)
        self.relu(x)
        x = self.linear2(x)
        self.dropout_1(x)
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


class SpatialDropout(nn.Module):
    def __init__(self):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(inplace=True, p=0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x


class TimeModel(nn.Module):
    def __init__(
        self,
        num_neurons: int = 16,
        num_channels: int = 2,
    ):
        super(TimeModel, self).__init__()
        self.num_neurons = num_neurons
        self.conv_input = nn.Conv1d(
            num_channels, self.num_neurons, (5,), padding="valid"
        )
        self.conv_16_32 = nn.Conv1d(
            self.num_neurons, self.num_neurons * 2, (5,), padding="valid"
        )
        self.conv_32 = nn.Conv1d(
            self.num_neurons * 2, self.num_neurons * 2, (3,), padding="valid"
        )
        self.conv_32_256 = nn.Conv1d(
            self.num_neurons * 2, self.num_neurons**2, (3,), padding="valid"
        )
        self.conv_256 = nn.Conv1d(
            self.num_neurons**2, self.num_neurons**2, (3,), padding="valid"
        )
        self.spatial_dropout = SpatialDropout()
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(2)
        self.output = nn.Linear(256 * 371, 64)

    def forward(self, x):
        x = self.conv_input(x)
        self.relu(x)
        x = self.conv_16_32(x)
        self.relu(x)
        x = self.max_pool(x)
        x = self.spatial_dropout(x)
        print(x.shape)

        x = self.conv_32(x)
        self.relu(x)
        x = self.conv_32(x)
        self.relu(x)
        x = self.max_pool(x)
        x = self.spatial_dropout(x)

        print(x.shape)

        x = self.conv_32_256(x)
        self.relu(x)
        x = self.conv_256(x)
        self.relu(x)
        x = self.max_pool(x)
        self.dropout(x)
        print(x.shape)

        bsz, nch, time = x.shape
        x = x.view(bsz, -1)
        print(x.shape)
        x = self.output(x)
        self.relu(x)
        self.dropout(x)
        return x


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and time steps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class Pt5Model(nn.Module):
    """
    Implementation of the model used by the work package 5: time distributed
    """

    def __init__(
        self,
        data_t_steps: int,
        num_input_channels: int = 2,
        num_neurons: int = 32,
        num_classes: int = 5,
    ):
        super(Pt5Model, self).__init__()
        self.num_neurons = num_neurons
        self.conv_block = TimeModel(num_neurons, num_input_channels)
        self.conv = nn.Conv1d(
            self.num_neurons**2, self.num_neurons**2, (3,), padding="same"
        )
        self.conv_output = nn.Conv1d(
            self.num_neurons**2, num_classes, (3,), padding="same"
        )
        self.relu = nn.ReLU(inplace=True)
        self.spatial_dropout = SpatialDropout()
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(inplace=True, p=0.05)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.spatial_dropout(x)
        x = self.conv(x)
        self.dropout_1(x)
        x = self.conv(x)
        y = self.conv_output(x)
        return y

    def freeze_all_but_last(self):
        # named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if "output" in n or "linear3" in n:
                pass
            else:
                p.requires_grad = False


# TODO: Implement AttSleepModel

# TODO: Implement an S4 model

# TODO: Implement a mamba model
"""
https://blog.oxen.ai/mamba-linear-time-sequence-modeling-with-selective-state-spaces-arxiv-dives/
https://www.reddit.com/r/MachineLearning/comments/18iph6a/d_can_someone_describe_how_the_ssm_in_mamba_is/
https://www.reddit.com/r/learnmachinelearning/comments/190grty/mamba_and_s4_explained_architecture_parallel_scan/
https://github.com/state-spaces/mamba
https://github.com/johnma2006/mamba-minimal
https://www.catalyzex.com/paper/arxiv:2312.00752/code
https://arxiv.org/abs/2312.00752
https://www.youtube.com/watch?v=OpJMn8T7Z34
https://www.youtube.com/watch?v=9dSkvxS2EB0&t=31s
https://www.youtube.com/watch?v=dKJEpOtVgXc&t=172s
https://www.youtube.com/watch?v=8Q_tqwpTpVU&t=22s
"""

if __name__ == "__main__":
    signal_frequency = 100
    window_size = 30
    test_resnet = False
    test_pt5_model = True
    steps_sequence = window_size * signal_frequency
    t = torch.rand((1, 2, steps_sequence))  # num_series, num_channels, seq_length
    if test_resnet:
        num_cnn_blocks = 8
        m = ResnetSignalClassificationEEG(
            steps_sequence, num_conv_blocks=num_cnn_blocks
        )
        tes = m(t)
        print(tes.shape)
    if test_pt5_model:
        tm = TimeModel()
        m = TimeDistributed(tm, batch_first=True)
        m(t)
        # m = Pt5Model(t.shape[2], num_input_channels=t.shape[1], num_neurons=16, num_classes=5)
        # tes = m(t)
        print(tes.shape)
