import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ai4ha.layers import PositionalEncoding, Patchify
from einops import rearrange, repeat, reduce
# from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


def ttsgan_discriminator_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "TTSGAN"
    name += f"_p{config['discriminator']['params']['patch_size']}"
    name += f"_d{config['discriminator']['params']['data_embed_dim']}"
    name += f"_ce{config['discriminator']['params']['class_embed_dim']}"
    name += f"_n{config['discriminator']['params']['n_layers']}"
    name += f"_dr{config['discriminator']['params']['dropout_rate']}"
    name += f"_h{config['discriminator']['params']['num_heads']}"
    if 'pos_encodings' in config['discriminator']['params']:
        name += f"_p{config['discriminator']['params']['pos_encodings']}"
    name += f"_cl{config['discriminator']['params']['class_logits']}"
    return name


class TTSGANDiscriminator(nn.Module):
    """_TTSGANDiscriminator model_

       class_logits = classtoken: only the class token is used for classification
                    = avpoolin: average pooling
    """

    def __init__(
        self,
        seq_length=512,
        seq_channels=3,
        n_classes=9,
        patch_size=16,
        data_embed_dim=10,
        class_embed_dim=10,
        n_layers=3,
        num_heads=5,
        dropout_rate=0.5,
        pos_encodings=True,
        class_logits='classtoken'
    ):
        super(TTSGANDiscriminator, self).__init__()
        self.seq_len = seq_length
        self.seq_channels = seq_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = class_embed_dim
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.pos_encodings = pos_encodings
        self.class_logits = class_logits

        # Reduces the input time series into a set of tokens
        self.patchify = Patchify(self.seq_channels, self.data_embed_dim,
                                 self.patch_size)
        # Adds positional encoding to the tokens
        self.pos_embedding = PositionalEncoding(self.data_embed_dim,
                                                self.dropout_rate)
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, data_embed_dim))

        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.data_embed_dim,
            nhead=self.num_heads,
            activation=nn.GELU(),
            norm_first=True,
            dropout=self.dropout_rate,
        )

        # Transformer encoder
        self.blocks = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.n_layers)

        # Classification head (logit)
        self.class_head = nn.Sequential(nn.LayerNorm(self.data_embed_dim),
                                        nn.Linear(self.data_embed_dim, 1))

        # Projection head
        self.proj_disc = nn.Linear(n_classes, data_embed_dim)

    def forward(self, x, labels):
        batch = x.shape[0]
        # convert to tokens
        x = self.patchify(x)
        x = rearrange(x, 'b e t -> b t e')

        # add class token
        cls_tokens = repeat(self.cls_token, "() t e -> b t e", b=batch)

        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding
        x = rearrange(x, 'b t e -> t b e')
        if self.pos_encodings:
            x = self.pos_embedding(x)
        # transformer
        x = self.blocks(x)

        # Only the class token is used for classification
        if self.class_logits == 'classtoken':
            x = x[0]
        # Average pooling
        elif self.class_logits == 'avpooling':
            x = reduce(x, "t b e -> b e", reduction="mean")

        if labels is not None:
            ohe = F.one_hot(labels, self.n_classes).float()
            proy = torch.sum(x * self.proj_disc(ohe), axis=1, keepdims=True)

            logit = self.class_head(x) + proy
        else:
            logit = self.class_head(x)
        return logit
