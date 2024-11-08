# Implementation of the WaveGAN model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.utils.data
import numpy as np


class PhaseShuffle(nn.Module):
    """
    Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8

    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary

    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """

    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            # Make sure to use PyTorcTrueh to generate number RNG state is all shared
            k = int(torch.Tensor(1).random_(
                0, 2 * self.shift_factor + 1)) - self.shift_factor

            # Return if no phase shift
            if k == 0:
                return x

            # Slice feature dimension
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)

            # Reflection padding
            x_shuffle = F.pad(x_trunc, pad, mode='reflect')

        else:
            # Generate shifts for each sample in the batch
            k_list = torch.Tensor(x.shape[0]).random_(0, 2*self.shift_factor+1)\
                - self.shift_factor
            k_list = k_list.numpy().astype(int)

            # Combine sample indices into lists so that less shuffle operations
            # need to be performed
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)

            # Make a copy of x for our output
            x_shuffle = x.clone()

            # Apply shuffle to each sample
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0),
                                            mode='reflect')
                else:
                    x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k),
                                            mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(
            x_shuffle.shape, x.shape)
        return x_shuffle


class WaveGANGenerator(nn.Module):
    """_WaveGAN generator with deconvolutions_

    Applies a succesion of deconvolutions to the input latent vector to 
    generate a sequence that has the specified output size assuming a latent
    vector that is transformed to the initial shape of the deconvolutions 
    (init_channels * init_dim).

    The number of deconvolutions is determined by the output size and the 
    initial shape. The first sequence of deconvolutions increase the size of
    the input to the output size reducing the number of channels in each layer.
    The layers apply batch normalization and ReLU activation functions.

    The output of the last deconvolution is the generated sequence.

    Assumes that the output size is a power of 2.
    Padding is calculated so each deconvolution doubles the size.

    TODO: Add support for upsamplings larger than 2, check that the sizes are
    coherent and the stride is correct.
    """

    def __init__(
        self,
        latent_dim=100,
        init_dim=16,
        init_channels=256,
        output_size=512,
        output_channels=8,
        kernel_size=16,
        stride=2,
        n_classes=0,
        # padding=1,
        batch_norm=True,
    ):
        super(WaveGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = output_channels
        self.init_dim = init_dim
        self.init_channels = init_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_norm = batch_norm
        self.n_classes = n_classes

        if kernel_size % 2 == 1:
            raise ValueError("The kernel size must be even")

        # Padding calculated so each deconvolution doubles the size
        self.padding = (kernel_size // 2) - 1

        # TODO: Add support for upsamplings larger than 2
        self.deconv_layers = int(np.log2(output_size // init_dim)) - 1

        # Transform the latent to the initial shape
        self.fc1 = nn.Linear(latent_dim + self.n_classes,
                             init_channels * init_dim)

        # Sequence of upsampling deconvolutions
        self.deconvolutions = []

        # Each block has a deconvolution, batch normalization and ReLU (if batch_norm is True)
        # The number of channels is halved in each block (assuming stride=2)
        for i in range(self.deconv_layers):
            deconv = nn.ConvTranspose1d(
                init_channels,
                init_channels // 2,
                kernel_size,
                stride,
                self.padding,
            )
            self.deconvolutions.append(deconv)
            if batch_norm:
                self.deconvolutions.append(nn.BatchNorm1d(init_channels // 2))
            self.deconvolutions.append(nn.ReLU())
            init_channels = init_channels // 2

        self.deconvolutions = nn.ModuleList(self.deconvolutions)

        # Output deconvolution maps to the output size and the output channels
        self.deconv_out = nn.ConvTranspose1d(
            init_channels,
            output_channels,
            kernel_size,
            stride,
            self.padding,
        )

    def forward(self, x, labels=None):
        if labels is not None:
            ohe = F.one_hot(labels, self.n_classes)
            x = torch.cat([x, ohe], dim=1)

        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.init_dim)
        for layer in self.deconvolutions:
            x = layer(x)
        x = self.deconv_out(x)
        return x


class Pulse2PulseGANGenerator(nn.Module):
    """ _Pulse2Pulse generator with deconvolutions_

        Uses a U-Net style architectures with convolutions for downsampling and
        deconvolutions for upsampling. The architecture is designed to generate
        sequences of the same size as the input sequences.
    """

    def __init__(
        self,
        seq_length=512,
        seq_channels=32,
        init_channels=32,
        kernel_size=16,
        stride=2,
        n_layers=3,
        n_classes=0,
        batch_norm=True,
    ):
        super(Pulse2PulseGANGenerator, self).__init__()
        self.seq_length = seq_length
        self.seq_channels = seq_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_norm = batch_norm
        self.n_classes = n_classes

        if kernel_size % 2 == 1:
            raise ValueError("The kernel size must be even")

        # Padding calculated so each deconvolution doubles the size
        self.padding = (kernel_size // 2) - 1

        self.n_layers = n_layers
        # Linear layer to adapt the class OHE to the latent space
        self.class_adapt = nn.Linear(self.n_classes, seq_length)
        class_cond = 1 if n_classes > 0 else 0

        self.convolutions = []
        # Fist convolution to adapt from the initial channels to the convolutions
        block = []
        block.append(
            nn.Conv1d(seq_channels + class_cond,
                      init_channels,
                      kernel_size,
                      stride,
                      padding=self.padding))
        if batch_norm:
            block.append(nn.BatchNorm1d(init_channels))
        block.append(nn.ReLU())
        self.convolutions.append(nn.Sequential(*block))

        # Succesive convolutions to downsample the input and doubling the channels
        for i in range(n_layers):
            block = []
            block.append(
                nn.Conv1d(init_channels * (2**i),
                          init_channels * (2**(i + 1)),
                          kernel_size,
                          stride,
                          padding=self.padding))
            if batch_norm:
                block.append(nn.BatchNorm1d(init_channels * (2**(i + 1))))
            block.append(nn.ReLU())
            self.convolutions.append(nn.Sequential(*block))
        self.convolutions = nn.ModuleList(self.convolutions)
        self.deconvolutions = []

        # Succesive deconvolutions to upsample the input and halving the channels
        block = []
        block.append(
            nn.ConvTranspose1d(init_channels * 2**(n_layers),
                               init_channels * 2**(n_layers - 1),
                               kernel_size,
                               stride,
                               padding=self.padding))
        self.first_deconvolution = nn.Sequential(*block)
        self.deconvolution = []
        for i in range(n_layers - 1):
            block = []
            block.append(
                nn.ConvTranspose1d(
                    init_channels * (2**(n_layers - i)),
                    init_channels * (2**(n_layers - i - 2)),
                    kernel_size,
                    stride,
                    padding=self.padding,
                ))
            if batch_norm:
                block.append(
                    nn.BatchNorm1d(init_channels * (2**(n_layers - i - 2))))
            block.append(nn.ReLU())
            self.deconvolutions.append(nn.Sequential(*block))
        self.deconvolutions = nn.ModuleList(self.deconvolutions)
        # Output deconvolution maps to the output size and the output channels
        block = []
        block.append(
            nn.ConvTranspose1d(init_channels * 2**(n_layers - i - 2),
                               seq_channels,
                               kernel_size,
                               stride,
                               padding=self.padding))
        self.out_deconvolution = nn.Sequential(*block)

    def forward(self, x, labels=None):
        # Conditioning adding the class information as an additional channel
        if labels is not None:
            ohe = F.one_hot(labels, self.n_classes)
            x = torch.cat([x, self.class_adapt(ohe.float()).unsqueeze(1)], dim=1)
        # apply convolutions
        skips = []
        for layer in self.convolutions:
            x = layer(x)
            skips.append(x)
        # apply deconvolutions
        x = self.first_deconvolution(skips.pop(-1))
        for layer in self.deconvolutions:
            cnv = skips.pop(-1)
            x = torch.cat([x, cnv], dim=1)
            x = layer(x)
        # output deconvolution
        x = self.out_deconvolution(x)
        return torch.sigmoid(x)


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
        padding=1,
        shift_factor=2,
        n_classes=0,
        sp_norm=False,
    ):
        super(WaveGANDiscriminator, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
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
                        nn.Conv1d(input_channels,
                                  init_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)))
            else:
                self.convolutions.append(
                    nn.Conv1d(input_channels,
                              init_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding))
            if shift_factor > 0:
                self.convolutions.append(PhaseShuffle(shift_factor))
            self.convolutions.append(nn.LeakyReLU(0.2))
            input_channels = init_channels
            init_channels = init_channels * 2
        self.convolutions = nn.ModuleList(self.convolutions)

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
