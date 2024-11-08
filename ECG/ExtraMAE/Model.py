import torch.nn as nn
import torch
import numpy as np
from util import random_indexes, take_indexes
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class Patchify(nn.Module):
    """_Transform a time series into a set of tokens via linear projection_

        Applies 1D convolution to the input time series, followed by a linear
        projection to obtain the tokenized representation of the time series.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, embed_dimension, patch_size, normalization=None):
        super(Patchify, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.activation = nn.GELU()
        self.normalization = normalization

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        # if self.normalization is not None:
        #     x = self.normalization(x)
        return x


class ExtraMAEEncoder(nn.Module):
    """_Encoder for the ExtraMAE model_

    Computes the patches of the series, selects a subset of them and applies
    the transformer encoder to the selected patches. Returns the selected
    patches and the indexes to reconstruct the original series.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels,
        series_length,
        embed_dimension,
        patch_size,
        mask_percent,
        num_layers,
        num_heads,
        norm_layer=nn.LayerNorm,
    ):
        super(ExtraMAEEncoder, self).__init__()
        self.series_length = series_length
        self.patch_size = patch_size
        self.percent = int((series_length // patch_size) * (1 - mask_percent))
        self.patchify = Patchify(in_channels, embed_dimension, patch_size,
                                 norm_layer)
        self.pos_embedding = nn.Parameter(
            torch.zeros((series_length // patch_size), 1, embed_dimension))
        self.tre_layer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dimension, nhead=num_heads),
            num_layers=num_layers,
        )

        self.norm = norm_layer(embed_dimension)

    def forward(self, x):
        batch = x.shape[0]
        indexes = [
            random_indexes(self.series_length // self.patch_size)
            for _ in range(batch)
        ]
        # Indexes to select the patches
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes],
                                                   axis=-1),
                                          dtype=torch.long).to(x.device)
        # Indexes to reconstruct the original series
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes],
                                                    axis=-1),
                                           dtype=torch.long).to(x.device)
        # Obtain the patches applying 1D convolution
        x = self.patchify(x)
        # The transformer expects the sequence length (t) to be the first dimension
        x = rearrange(x, "b c t -> t b c")
        x += self.pos_embedding
        # Rearranges the sequence
        x = take_indexes(x, forward_indexes)
        # Select a subset of the patches (rest is masked out)
        x = x[:self.percent]
        x = self.tre_layer(x)
        # Normalize the output
        if self.norm is not None:
            x = self.norm(x)
        return x, backward_indexes


class ExtraMAEDecoder(nn.Module):
    """_Decoder for the ExtraMAE model_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels,
        series_length,
        embed_dimension,
        patch_size,
        mask_percent,
        num_layers,
        num_heads,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(ExtraMAEDecoder, self).__init__()
        self.series_length = series_length
        self.patch_size = patch_size
        self.percent = int((series_length // patch_size) * mask_percent)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dimension))
        self.pos_embedding_d = torch.nn.Parameter(
            torch.zeros((series_length // patch_size), 1, embed_dimension)
        )

        self.trd_layer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dimension, nhead=num_heads),
            num_layers=num_layers,
        )
        self.head = torch.nn.Linear(embed_dimension, in_channels * patch_size)
        self.patch2img = Rearrange(
            "l b (c p1) -> b c (l p1)", p1=patch_size, l=series_length // patch_size
        )
        # self.norm = norm_layer(embed_dimension)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)

        features = features + self.pos_embedding_d
        features = self.trd_layer(features)
        features = self.head(features)

        mask = torch.zeros_like(features)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes)
        features = self.patch2img(features)
        mask = self.patch2img(mask)
        return features, mask


class ExtraMAE(nn.Module):
    """_ExtraMAE model_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels,
        series_length,
        embed_dimension,
        patch_size,
        mask_percent,
        num_layers,
        num_heads,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(ExtraMAE, self).__init__()
        self.encoder = ExtraMAEEncoder(
            in_channels,
            series_length,
            embed_dimension,
            patch_size,
            mask_percent,
            num_layers,
            num_heads,
            norm_layer,
        )
        self.decoder = ExtraMAEDecoder(
            in_channels,
            series_length,
            embed_dimension,
            patch_size,
            mask_percent,
            num_layers,
            num_heads,
            act_layer,
            norm_layer,
        )

    def forward(self, x):
        features, backward_indexes = self.encoder(x)
        return self.decoder(features, backward_indexes)

    def impute_one(self, x, mask):
        """_Imputes from a time series the masked parts_

        With one time series and a mask indicating the patches of the series
        that will be imputed, the model returns the imputed series.

        This does not work with batches of time series since the masks can have
        different lengths.

        Args:
            x (_type_): _time series_
            mask (_type_): _mask indicating the patches of the series that will be imputed_

        Returns:
            _type_: _description_
        """
        # Select the indexes of the patches that are not masked
        indexes = torch.nonzero(mask - 1, as_tuple=True)
        # Select the indexes of the patches that are masked (the ones to impute)
        imputed_indexes = torch.nonzero(mask, as_tuple=True)
        reordered_indexes = torch.cat((indexes[0], imputed_indexes[0]), dim=0)
        backward_indexes = np.argsort(reordered_indexes)
        reordered_indexes = reordered_indexes.unsqueeze(0).T
        backward_indexes = backward_indexes.unsqueeze(0).T

        # apply the encoder to the series
        series = self.encoder.patchify(x)
        series = rearrange(series, "b c t -> t b c")
        series += self.encoder.pos_embedding
        series = take_indexes(series, reordered_indexes.to(x.device))
        # Take the indexes of the patches that are not masked
        series = series[: indexes[0].shape[0]]
        series = self.encoder.tre_layer(series)

        # Expand the series to the length of the original series
        features = torch.cat(
            [
                series,
                self.decoder.mask_token.expand(
                    reordered_indexes.shape[0] - indexes[0].shape[0],
                    series.shape[1],
                    -1,
                ),
            ],
            dim=0,
        )

        # Apply the decoder to the features
        features = take_indexes(features, backward_indexes.to(x.device))
        features = features + self.decoder.pos_embedding_d
        features = self.decoder.trd_layer(features)
        features = self.decoder.head(features)
        features = self.decoder.patch2img(features)

        return features
