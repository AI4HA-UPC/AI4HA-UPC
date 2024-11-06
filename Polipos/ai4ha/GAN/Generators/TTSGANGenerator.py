import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ai4ha.layers import PositionalEncoding
from einops import rearrange


def ttsgan_generator_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "TTSGAN"
    name += f"_ld{config['generator']['params']['latent_dim']}"
    name += f"_de{config['generator']['params']['data_embed_dim']}"
    if config['generator']['params']['n_classes'] > 0:
        if config['generator']['params']['class_embed_dim'] > 0:
            name += f"_ce{config['generator']['params']['class_embed_dim']}"
    name += f"_l{config['generator']['params']['n_layers']}"
    name += f"_h{config['generator']['params']['num_heads']}"
    name += f"_dr{config['generator']['params']['dropout_rate']}"
    if 'pos_encodings' in config['generator']['params']:
        name += f"_p{config['generator']['params']['pos_encodings']}"
    return name


class TTSGANGenerator(nn.Module):

    def __init__(
        self,
        latent_dim=100,
        seq_length=150,
        seq_channels=3,
        n_classes=9,
        data_embed_dim=10,
        class_embed_dim=10,
        n_layers=3,
        num_heads=5,
        dropout_rate=0.5,
        pos_encodings=False,
    ):
        super(TTSGANGenerator, self).__init__()
        self.seq_len = seq_length
        self.seq_channels = seq_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.class_embed_dim = class_embed_dim
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pos_encodings = pos_encodings

        self.l1 = nn.Linear(self.latent_dim + self.class_embed_dim,
                            self.seq_len * self.data_embed_dim)
        self.class_embedding = nn.Embedding(self.n_classes,
                                            self.class_embed_dim)

        self.pos_encoder = PositionalEncoding(self.data_embed_dim,
                                              self.dropout_rate)

        encoder_layer = TransformerEncoderLayer(
            self.data_embed_dim,
            self.num_heads,
            dim_feedforward=2048,  # Default value
            activation=nn.GELU(),
            norm_first=True,
            dropout=self.dropout_rate)

        self.blocks = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_layers,
        )

        self.deconv = nn.Sequential(
            nn.Conv1d(self.data_embed_dim, self.seq_channels, 1, 1, 0))

    def forward(self, z, labels):
        c = self.class_embedding(labels)
        x = torch.cat([z, c], 1)
        x = self.l1(x)
        x = x.view(-1, self.seq_len, self.data_embed_dim)
        x = rearrange(x, "b t e -> t b e")
        if self.pos_encodings:
            x = self.pos_encoder(x)
        x = self.blocks(x)
        x = rearrange(x, "t b e -> b e t")
        output = self.deconv(x)
        return output
