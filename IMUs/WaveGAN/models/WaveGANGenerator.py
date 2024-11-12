import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np


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

    def __init__(self,
                 latent_dim=100,
                 init_dim=16,
                 init_channels=256,
                 output_size=512,
                 output_channels=8,
                 kernel_size=16,
                 stride=2,
                 n_classes=0,
                 class_embedding_dim=0,
                 normalization=None,
                 upsampling=False):
        super(WaveGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = output_channels
        self.init_dim = init_dim
        self.init_channels = init_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalization = normalization
        self.n_classes = n_classes

        if kernel_size % 2 == 1 and not upsampling:
            raise ValueError("The kernel size must be even")

        if kernel_size % 2 == 1 and upsampling:
            raise ValueError("The kernel size must be odd")

        # Padding calculated so each deconvolution doubles the size
        self.padding = (kernel_size // 2) - 1

        # TODO: Add support for upsamplings larger than 2
        self.deconv_layers = int(np.log2(output_size // init_dim)) - 1

        # When conditioning, if class_embedding is 0, the class is one-hot encoded
        # else an embedding is used to transform the class to the latent space
        if n_classes > 0:
            if class_embedding_dim > 0:
                self.class_embedding = nn.Embedding(self.n_classes,
                                                    class_embedding_dim)
                class_dim = class_embedding_dim
            else:
                self.class_embedding = None
                class_dim = n_classes
        else:
            class_dim = 0
        # Transform the latent to the initial shape
        self.fc1 = nn.Linear(latent_dim + class_dim, init_channels * init_dim)

        # Sequence of upsampling deconvolutions
        self.deconvolutions = []

        # Each block has a deconvolution, batch/instance normalization and ReLU
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
            if normalization == 'batch':
                self.deconvolutions.append(nn.BatchNorm1d(init_channels // 2))
            elif normalization == 'instance':
                self.deconvolutions.append(
                    nn.InstanceNorm1d(init_channels // 2))
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
        # Concatenates the class information to the latent vector
        if labels is not None:
            if self.class_embedding is not None:
                x = torch.cat([x, self.class_embedding(labels)], dim=1)
            else:
            	ohe = F.one_hot(labels, self.n_classes)
            	x = torch.cat([x, ohe], dim=1)

        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.init_dim)
        for layer in self.deconvolutions:
            x = layer(x)
        x = self.deconv_out(x)
        return x

