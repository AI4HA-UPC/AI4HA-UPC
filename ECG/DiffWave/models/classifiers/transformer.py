import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.classifiers.layers.Layers import PositionalEncoding, Patchify
from einops import rearrange, repeat, reduce


def transformer_classifier_exp_name(config):
    """Generates a name for the experiment based on the configuration"""
    name = "TRANSCLASS"
    name += f"_p{config['model']['params']['patch_size']}"
    name += f"_d{config['model']['params']['data_embed_dim']}"
    name += f"_n{config['model']['params']['n_layers']}"
    name += f"_dr{config['model']['params']['dropout_rate']}"
    name += f"_h{config['model']['params']['num_heads']}"
    if "pos_encodings" in config["model"]["params"]:
        name += f"_p{config['model']['params']['pos_encodings']}"
    name += f"_cl{config['model']['params']['class_logits']}"
    return name


class TransformerClassifier(nn.Module):
    """_TransformerClassifier model_

    Mostly copied from TTSGANDiscriminator

    class_logits = classtoken: only the class token is used for
                   classification
                 = avgpool: average pooling
    """

    def __init__(
        self,
        seq_length=512,
        seq_channels=3,
        n_classes=9,
        patch_size=100,
        data_embed_dim=128,
        n_layers=8,
        n_heads=16,
        dropout_rate=0.25,
        pos_encodings=True,
        class_logits="classtoken",
    ):
        super(TransformerClassifier, self).__init__()
        self.seq_len = seq_length
        self.seq_channels = seq_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.data_embed_dim = data_embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.pos_encodings = pos_encodings
        self.class_logits = class_logits

        # Reduces the input time series into a set of tokens
        self.patchify = Patchify(
            self.seq_channels, self.data_embed_dim, self.patch_size
        )
        # Adds positional encoding to the tokens
        self.pos_embedding = PositionalEncoding(self.data_embed_dim, self.dropout_rate)
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.data_embed_dim))

        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.data_embed_dim,
            nhead=self.n_heads,
            activation=nn.GELU(),
            norm_first=True,
            dropout=self.dropout_rate,
        )

        # Transformer encoder
        self.blocks = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.n_layers
        )

        # Classification head (logits)
        self.class_head = nn.Sequential(
            nn.LayerNorm(self.data_embed_dim),
            nn.Linear(self.data_embed_dim, n_classes),
            # nn.LogSoftmax()
        )

    def forward(self, x):
        batch = x.shape[0]
        # convert to tokens
        x = self.patchify(x)
        x = rearrange(x, "b e t -> b t e")

        # add class token
        cls_tokens = repeat(self.cls_token, "() t e -> b t e", b=batch)

        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding
        x = rearrange(x, "b t e -> t b e")
        if self.pos_encodings:
            x = self.pos_embedding(x)
        # transformer
        x = self.blocks(x)

        # Only the class token is used for classification
        if self.class_logits == "classtoken":
            x = x[0]
        # Average pooling
        elif self.class_logits == "avgpool":
            x = reduce(x, "t b e -> b e", reduction="mean")

        logit = self.class_head(x)
        return logit


if __name__ == "__main__":
    width_sequence = 1000
    depth_sequence = 1
    num_channels = 1
    height_sequence = 12
    num_classes = 7
    data = torch.randn((2, height_sequence, width_sequence))
    model = TransformerClassifier(
        seq_length=width_sequence, seq_channels=height_sequence, n_classes=num_classes
    )
    y = model(data)
    print(y)
