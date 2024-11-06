import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.utils.data
from ai4ha.layers import PhaseShuffle


def wavegan_discriminator_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "WaveGAN"
    name += f"_n{config['discriminator']['params']['n_layers']}"
    name += f"_c{config['discriminator']['params']['init_channels']}"
    name += f"_k{config['discriminator']['params']['kernel_size']}"
    name += f"_s{config['discriminator']['params']['stride']}"
    name += f"_sf{config['discriminator']['params']['shift_factor']}"
    name += f"_sp{config['discriminator']['params']['sp_norm']}"
    return name


class WaveGANDiscriminator(nn.Module):
    """_WaveGAN discriminator with convolutions_

    Applies a succession of convolutions to the input sequence to
    generate a scalar that represents the probability that the input
    is real.

    The number of convolutions is determined by the input size and the
    output size. The first sequence of convolutions decrease the size of
    the input to the output size increasing the number of channels in each layer.
    The layers apply batch normalization and LeakyReLU activation functions.

    The output of the last convolution is the scalar.

    Assumes that the output size is a power of 2.
    """

    def __init__(
        self,
        input_size=512,
        input_channels=1,
        n_layers=3,
        init_channels=16,
        kernel_size=16,
        stride=2,
        shift_factor=2,
        n_classes=0,
        sp_norm=False,
    ):
        super(WaveGANDiscriminator, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) - 1
        self.shift_factor = shift_factor
        self.n_classes = n_classes

        if input_size % (2**n_layers) != 0:
            raise ValueError(
                "The input size must be a multiple of 2^num_layers")
        if input_size < 2**n_layers:
            raise ValueError("The input size must be larger than 2^num_layers")

        if n_classes == 1:
            raise ValueError(
                "The number of classes must be larger than 1 when is a conditional model"
            )
        # Sequence of downsampling convolutions doubling the number of channels
        self.convolutions = []
        self.num_channels = input_channels
        for i in range(n_layers):
            if sp_norm:
                self.convolutions.append(
                    spectral_norm(
                        nn.Conv1d(
                            input_channels,
                            init_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=self.padding,
                        )))
            else:
                self.convolutions.append(
                    nn.Conv1d(
                        input_channels,
                        init_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=self.padding,
                    ))
            if shift_factor > 0:
                self.convolutions.append(PhaseShuffle(shift_factor))
            self.convolutions.append(nn.LeakyReLU(0.2))
            input_channels = init_channels
            init_channels = init_channels * 2
        self.convolutions = nn.ModuleList(self.convolutions)

        # Projector discriminator
        if self.n_classes > 0:
            if sp_norm:
                self.proj_disc = spectral_norm(
                    nn.Linear(n_classes,
                              input_channels * (input_size // (2**n_layers))))
            else:
                self.proj_disc = nn.Linear(
                    n_classes, input_channels * (input_size // (2**n_layers)))

        # output layer
        if sp_norm:
            self.fc = spectral_norm(
                nn.Linear(input_channels * (input_size // (2**n_layers)), 1))
        else:
            self.fc = nn.Linear(input_channels * (input_size // (2**n_layers)),
                                1)

    def forward(self, x, labels=None):
        for layer in self.convolutions:
            x = layer(x)
        x = x.view(x.size(0), -1)
        if labels is not None:
            ohe = F.one_hot(labels, self.n_classes).float()
            proy = torch.sum(x * self.proj_disc(ohe), axis=1, keepdims=True)

            x = self.fc(x) + proy
        else:
            x = self.fc(x)
        # return torch.sigmoid(x)
        return x
