import torch.nn as nn
import torch
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from models.Layers import Patchify


def random_indexes(size):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0,
                        repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


def extra_masked_autoencoder_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "ExtraMAE"
    name += f"_l{config['model']['layers']}"
    name += f"_h{config['model']['heads']}"
    name += f"_p{config['model']['patch_size']}"
    name += f"_m{config['model']['mask_percent']}"
    name += f"_e{config['model']['embed_dim']}"
    return name


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
        #mask_percent = random.randint(0.2, 0.8)
        mask_percent = np.random.uniform(0.2, 0.8)
        self.percent = int((self.series_length // self.patch_size) * (1 - mask_percent))        
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
        self.mask_token = torch.nn.Parameter(torch.zeros(
            1, 1, embed_dimension))
        self.pos_embedding_d = torch.nn.Parameter(
            torch.zeros((series_length // patch_size), 1, embed_dimension))

        self.trd_layer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dimension, nhead=num_heads),
            num_layers=num_layers,
        )
        self.head = torch.nn.Linear(embed_dimension, in_channels * patch_size)
        self.patch2img = Rearrange("l b (c p1) -> b c (l p1)",
                                   p1=patch_size,
                                   l=series_length // patch_size)
        # self.norm = norm_layer(embed_dimension)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0],
                    features.shape[1], -1),
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


class ExtraMAEDiscriminator(nn.Module):
    """_Discriminator MAE for adversarial training_

    Transformer for computing the adversarial loss.
    Obtains the patches of the series, selects a subset of them and applies the
    transformer encoder to the selected patches. Return a logit for each patch
    determining if it is real or fake.
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
        super(ExtraMAEDiscriminator, self).__init__()
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
        self.head = nn.Linear(embed_dimension, 1)

    def forward(self, x):
        indexes = random_indexes(self.series_length // self.patch_size)
        forward_indexes = torch.as_tensor(indexes[0], dtype=torch.long).to(x.device)
        x = self.patchify(x)
        x = rearrange(x, "b c t -> t b c")
        x += self.pos_embedding
        x = take_indexes(x, forward_indexes)
        mask_percent = random.randint(0.2, 0.8)
        self.percent = int((series_length // patch_size) * (1 - mask_percent))       
        x = x[:self.percent]
        x = self.tre_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.head(x)
        return x


class ExtraMAE(nn.Module):
    """_ExtraMAE model_

    Masked autoencoder for time series imputation.

    Based on the original MAE model, this model uses a transformer to encode
    the patches of the series and a transformer to decode the patches. The
    decoder also outputs a mask indicating the patches that were imputed.

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
        backward_indexes = np.argsort(reordered_indexes.cpu())
        reordered_indexes = reordered_indexes.unsqueeze(0).T
        backward_indexes = backward_indexes.unsqueeze(0).T
        
        

        # apply the encoder to the series
        series = self.encoder.patchify(x)
        series = rearrange(series, "b c t -> t b c")
        series += self.encoder.pos_embedding
        series = take_indexes(series, reordered_indexes.to(x.device))
        
        if torch.any(reordered_indexes >= series.shape[0]) or torch.any(reordered_indexes < 0):
        	raise ValueError("Index out of bounds detected before calling take_indexes.")
        # Take the indexes of the patches that are not masked
        series = series[:indexes[0].shape[0]]
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
