import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import math


class UpsamplingConvo1D(nn.Module):
    """_Upsampling plus convolution as alternative deconvolution_

    Applies an upsampling layer to the input sequence and then applies a
    convolution to the upsampled sequence.

    This is an alternative to 1D deconvolution used in waveGAN/pulse2pulse.
    """

    def __init__(self, in_channels, out_channels, upsample, kernel_size,
                 stride, padding):
        super(UpsamplingConvo1D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode="nearest")
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        return x


class WaveGANDeconvolution(nn.Module):
    """_WaveGAN deconvolution layer_

    Applies an upsampling layer to the input sequence and then applies a
    convolution to the upsampled sequence.

    This is an alternative to 1D deconvolution used in waveGAN/pulse2pulse.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 upsampling=False):
        super(WaveGANDeconvolution, self).__init__()
        if upsampling:
            self.deconvo = UpsamplingConvo1D(in_channels, out_channels, stride,
                                             kernel_size, 1, 0)
        else:
            self.deconvo = nn.ConvTranspose1d(in_channels, out_channels,
                                              kernel_size, stride, padding)

    def forward(self, x):
        return self.deconvo(x)


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


class PositionalEncoding(nn.Module):
    """ __Positional encoding layer__

    Taken from torch tutorial on transformer:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html


    """

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Patchify(nn.Module):
    """_Transform a time series into a set of tokens via linear projection_

        Applies 1D convolution to the input time series, to obtain the tokenized 
        representation of the time series. This is a linear projection of the
        input time series considering all the information of the channels.

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_channels,
                 embed_dimension,
                 patch_size,
                 norm_layer=True):
        super(Patchify, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.activation = nn.GELU()
        # if norm_layer:
        #     self.norm = nn.LayerNorm((embed_dimension))
        # else:
        #     self.norm = None

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        # if self.norm is not None:
        #     x = self.norm(x)
        return x


class MixPatchify(nn.Module):
    """_Transform a time series into a set of tokens via 1-1D convolution_

        Applies kernel 1 1D convolution to the input time series to obtain
        half the embedded dimension and then obtains the patches applying
        a kernel patch_size 1D convolution.

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_channels,
                 embed_dimension,
                 patch_size,
                 norm_layer=True):
        super(MixPatchify, self).__init__()

        self.channel_mix = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dimension//2,
            kernel_size=1,
            stride=1,
        )

        self.conv1d = nn.Conv1d(
            in_channels=embed_dimension//2,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        # if norm_layer:
        #     self.norm1 = nn.LayerNorm(embed_dimension)
        #     self.norm2 = nn.LayerNorm(embed_dimension)
        # else:
        #     self.norm1 = None
        #     self.norm2 = None

    def forward(self, x):
        x = self.channel_mix(x)
        x = self.activation1(x)
        # if self.norm1 is not None:
        #     x = self.norm1(x)
        x = self.conv1d(x)
        x = self.activation2(x)
        # if self.norm2 is not None:
        #     x = self.norm2(x)
        return x


class PatchDiscriminator1D(nn.Module):
    """_Generates a logit for each patch of the series_

        Applies 1D convolution to the input time series to obtain
        patches, and uses a 1x1 convolutional
        layer to compute a logit for patches.

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 sequence_length,
                 in_channels,
                 embed_dimension,
                 patch_size,
                 normalization=None):
        super(PatchDiscriminator1D, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.output = nn.Sequential(
            nn.Conv1d(embed_dimension,
                      sequence_length // patch_size,
                      kernel_size=1,
                      stride=1), nn.Sigmoid())

        self.activation = nn.GELU()
        self.normalization = normalization

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.output(x)
        # if self.normalization is not None:
        #     x = self.normalization(x)
        return x
