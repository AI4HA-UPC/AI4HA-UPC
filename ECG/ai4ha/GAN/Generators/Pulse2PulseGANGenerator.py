import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from ai4ha.layers.Layers import WaveGANDeconvolution, UpsamplingConvo1D


def pulse2pulse_gan_generator_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "Pulse2PulseGAN"
    name += f"_i{config['generator']['params']['init_channels']}"
    name += f"_k{config['generator']['params']['kernel_size']}"
    name += f"_s{config['generator']['params']['stride']}"
    name += f"_n{config['generator']['params']['normalization']}"
    name += f"_l{config['generator']['params']['n_layers']}"
    name += f"_ce{config['generator']['params']['class_embedding']}"
    name += f"_n{config['generator']['params']['normalization']}"
    return name


class Pulse2PulseGANGenerator(nn.Module):
    """ _Pulse2Pulse generator with deconvolutions_

        Uses a U-Net style architecture with convolutions for downsampling and
        deconvolutions for upsampling. The architecture is designed to generate
        sequences of the same size as the input sequences.

        The input is a sequence of shape (batch, seq_channels, seq_length) and
        the output is a sequence of shape (batch, seq_channels, seq_length).

        seq_length: int - Length of the input and output sequences
        seq_channels: int - Number of channels in the input and output sequences
        init_channels: int - Number of channels in the first convolution
        kernel_size: int - Size of the kernel in the convolutions and
                            deconvolutions
        stride: int - Stride in the convolutions and deconvolutions
        n_layers: int - Number of layers in the U-Net (is has to be consistent 
                        with the length of the signal)        
        n_classes: int - Number of classes for the conditioning
        normalization: str - Type of normalization to use, can be 'batch' or
                            'instance'
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
        class_embedding=False,
        normalization=None,
    ):
        super(Pulse2PulseGANGenerator, self).__init__()
        self.seq_length = seq_length
        self.seq_channels = seq_channels
        self.init_channels = init_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalization = normalization
        self.n_classes = n_classes
        self.class_embedding = class_embedding

        if kernel_size % 2 == 1:
            raise ValueError("The kernel size must be even")

        # Padding calculated so each deconvolution doubles the size
        self.padding = (kernel_size // 2) - 1

        self.n_layers = n_layers
        # Layer to adapt the class to the latent space via embedding or linear
        # projection of the OHE
        if class_embedding:
            self.class_adapt = nn.Embedding(self.n_classes, seq_length)
        else:
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
        if normalization == 'batch':
            block.append(nn.BatchNorm1d(init_channels))
        elif normalization == 'instance':
            block.append(nn.InstanceNorm1d(init_channels))
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
            if normalization == 'batch':
                block.append(nn.BatchNorm1d(init_channels * (2**(i + 1))))
            elif normalization == 'instance':
                block.append(nn.InstanceNorm1d(init_channels * (2**(i + 1))))
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
            if normalization == 'batch':
                block.append(
                    nn.BatchNorm1d(init_channels * (2**(n_layers - i - 2))))
            elif normalization == 'instance':
                block.append(
                    nn.InstanceNorm1d(init_channels * (2**(n_layers - i - 2))))
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
            if self.n_classes == 0:
                raise ValueError("The generator was not initialized with the "
                                 "number of classes")
            if self.class_embedding:
                label_embed = self.class_adapt(labels)

            else:
                label_embed = self.class_adapt(
                    F.one_hot(labels, self.n_classes).float())

            x = torch.cat([x, label_embed.unsqueeze(1)], dim=1)
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
        return x
