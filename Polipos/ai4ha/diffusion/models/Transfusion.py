###
# Copied from https://github.com/fahim-sikder/TransFusion/blob/main/ddpm.py
###

from dataclasses import dataclass
import torch
from torch import nn
from ai4ha.layers.Layers import PositionalEncoding
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, GaussianFourierProjection


class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


@dataclass
class TransEncoderOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


# class TransEncoder2(ModelMixin, ConfigMixin):

#     @register_to_config
#     def __init__(self,
#                  sample_size,
#                  in_channels,
#                  latent_dim,
#                  num_heads,
#                  num_layers=6,
#                  dropout=0.1,
#                  activation='gelu',
#                  ff_size=1024):
#         super().__init__()
#         self.sample_size = sample_size
#         self.channels = in_channels
#         self.self_condition = None
#         self.latent_dim = latent_dim
#         self.num_heads = num_heads
#         self.ff_size = ff_size
#         self.activation = activation
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.pos_enc = PositionalEncoding(self.latent_dim)
#         self.emb_timestep = TimestepEmbedder(
#             self.latent_dim, PositionalEncoding(self.latent_dim))

#         # self.emb_timestep = TimestepEmbedding(self.channels, self.latent_dim)
#         self.input_dim = nn.Linear(self.channels, self.latent_dim)
#         self.output_dim = nn.Linear(self.latent_dim, self.channels)

#         self.TransEncLayer = nn.TransformerEncoderLayer(
#             d_model=self.latent_dim,
#             nhead=self.num_heads,
#             dim_feedforward=self.ff_size,
#             dropout=self.dropout,
#             activation=self.activation)

#         self.TransEncodeR = nn.TransformerEncoder(self.TransEncLayer,
#                                                   num_layers=self.num_layers)

#     def forward(self, sample, timestep, class_labels=None, return_dict=True):
#         x = torch.transpose(sample, 1, 2)
#         x = self.input_dim(x)
#         x = torch.transpose(x, 0, 1)
#         embed = self.emb_timestep(timestep)
#         time_added_data = torch.cat((embed, x), axis=0)
#         time_added_data = self.pos_enc(time_added_data)
#         trans_output = self.TransEncodeR(time_added_data)[1:]
#         final_output = self.output_dim(trans_output)
#         transposed_data = final_output.permute(1, 2, 0)
#         if not return_dict:
#             return (transposed_data, )

#         return TransEncoderOutput(sample=transposed_data)


class TransEncoder(ModelMixin, ConfigMixin):
    """_Transformer network for denoising diffusion_

    Args:
        ModelMixin (_type_): _description_
        ConfigMixin (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    @register_to_config
    def __init__(self,
                 sample_size,
                 in_channels,
                 latent_dim,
                 num_heads,
                 num_layers=6,
                 dropout=0.1,
                 activation='gelu',
                 ff_size=1024,
                 flip_sin_to_cos=True,
                 freq_shift=0.0,
                 time_embedding_type='positional',
                 num_class_embeds=0,
                 time_class_comb='concat'):
        super().__init__()
        self.sample_size = sample_size
        self.channels = in_channels
        self.self_condition = None
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.activation = activation
        self.num_layers = num_layers
        self.dropout = dropout
        self.time_class_comb = time_class_comb
        self.pos_enc = PositionalEncoding(self.latent_dim)
        if time_embedding_type == 'fourier':
            self.emb_timestep = GaussianFourierProjection(
                embedding_size=latent_dim,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos)
            timestep_input_dim = latent_dim * 2
        elif time_embedding_type == 'positional':
            self.emb_timestep = Timesteps(latent_dim, flip_sin_to_cos,
                                          freq_shift)
            timestep_input_dim = latent_dim
        self.time_mlp = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=latent_dim,
            act_fn='silu',
        )
        if num_class_embeds > 0:
            self.class_embed = nn.Embedding(num_class_embeds, latent_dim)
        else:
            self.class_embed = None

        self.input_dim = nn.Linear(self.channels, self.latent_dim)
        self.output_dim = nn.Linear(self.latent_dim, self.channels)

        self.TransEncLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)

        self.TransEncodeR = nn.TransformerEncoder(self.TransEncLayer,
                                                  num_layers=self.num_layers,
                                                  enable_nested_tensor=False)

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        x = torch.transpose(sample, 1, 2)
        x = self.input_dim(x)
        x = torch.transpose(x, 0, 1)

        # Time embedding
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] != x.shape[1]:
            timestep = timestep.expand(x.shape[1])
        embed = self.time_mlp(self.emb_timestep(timestep))

        # Class embedding
        if self.class_embed is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels must be provided if num_class_embeds > 0")
            if class_labels.ndim == 1:
                class_labels = class_labels.unsqueeze(1)
            class_embed = self.class_embed(class_labels)
            class_embed = class_embed.permute(1, 0, 2)

        # Time - class combination
        if self.time_class_comb == 'concat':
            time_added_data = torch.cat((embed.unsqueeze(0), x), axis=0)
            time_added_data = torch.cat((class_embed, time_added_data), axis=0)
        elif self.time_class_comb == 'add':
            time_added_data = torch.cat((embed.unsqueeze(0) + class_embed, x),
                                        axis=0)
        else:
            raise ValueError(
                f"Invalid time_class_comb: {self.time_class_comb}")

        time_added_data = self.pos_enc(time_added_data)

        if self.time_class_comb == 'concat':
            trans_output = self.TransEncodeR(time_added_data)[2:]
        elif self.time_class_comb == 'add':
            trans_output = self.TransEncodeR(time_added_data)[1:]
        final_output = self.output_dim(trans_output)
        transposed_data = final_output.permute(1, 2, 0)
        if not return_dict:
            return (transposed_data, )

        return TransEncoderOutput(sample=transposed_data)
