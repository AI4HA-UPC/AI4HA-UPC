# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @2023-04-19 Added class_labels parameter on forward method for class conditioned DDPM/DDIM
# @2023-04-19 Fixing class embedding for class conditioned DDPM/DDIM


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block, DownResnetBlock1D, Downsample1D, DownBlock1D, AttnDownBlock1D, DownBlock1DNoSkip, Downsample1d, ResConvBlock, UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip, Upsample1d
from diffusers.models.resnet import ResidualTemporalBlock1D
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.activations import get_activation

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from models.Layers import Patchify
import numpy as np


@dataclass
class CondUNet1DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor

     
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
     
              
class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_value_dim, out_dim, num_heads=2):
        super().__init__()
        self.projection_query = nn.Linear(query_dim, out_dim)
        self.projection_key = nn.Linear(key_value_dim, out_dim)
        self.projection_value = nn.Linear(key_value_dim, out_dim)
        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, attention_mask=None):
        query = self.projection_query(query)
        key = self.projection_key(key)
        value = self.projection_value(value)
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=attention_mask)
        return attn_output        
        
        
class CrossAttnDownBlock1D(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None, attention_head_dim: int = 1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        self.attention_head_dim = attention_head_dim
        self.num_heads = in_channels // self.attention_head_dim
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=2),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=2),  # Adjust dimensions
            CrossAttentionLayer(out_channels, 128, out_channels, num_heads=2),  # Adjust dimensions
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.down(hidden_states)

        for resnet, attn in zip(self.resnets, self.attentions):
            # attn          
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            hidden_states = resnet(hidden_states)
            
            # Ensure encoder_hidden_states has consistent shape
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)  # Fix permutation inconsistency

            hidden_states = hidden_states.permute(0, 2, 1)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            else:
                encoder_hidden_states = encoder_hidden_states
            
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)
            hidden_states = attn(hidden_states, encoder_hidden_states, encoder_hidden_states)     
            
            hidden_states = hidden_states.permute(0, 2, 1)

        return hidden_states, (hidden_states,)

DownBlockType = Union[DownResnetBlock1D, DownBlock1D, AttnDownBlock1D, DownBlock1DNoSkip, CrossAttnDownBlock1D, CrossAttnDownBlock1D]

def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
) -> DownBlockType:
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=in_channels)
    elif down_block_type == "CrossAttnDownBlock1D":
        return CrossAttnDownBlock1D(out_channels=out_channels, in_channels=in_channels)
    raise ValueError(f"{down_block_type} does not exist.")
    
    

UpBlockType = Union[UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip]
    
class CrossAttnUpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=2),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=2),  # Adjust dimensions
            CrossAttentionLayer(out_channels, 128, out_channels, num_heads=2),  # Adjust dimensions
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet, attn in zip(self.resnets, self.attentions):
            
            # attn          
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            hidden_states = resnet(hidden_states)
            
            # Ensure encoder_hidden_states has consistent shape
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)  # Fix permutation inconsistency

            hidden_states = hidden_states.permute(0, 2, 1)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            else:
                encoder_hidden_states = encoder_hidden_states
            
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)
            
            hidden_states = attn(hidden_states, encoder_hidden_states, encoder_hidden_states)     
            
            hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.up(hidden_states)

        return hidden_states


def get_up_block(
    up_block_type: str, num_layers: int, in_channels: int, out_channels: int, temb_channels: int, add_upsample: bool
) -> UpBlockType:
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=out_channels)
    elif up_block_type == "CrossAttnUpBlock1D":
        return CrossAttnUpBlock1D(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock1DSimpleCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        skip_time_act: bool = False,
        only_cross_attention: bool = False,
        cross_attention_norm: Optional[str] = None,
        out_channels: int = 0,
        mid_channels: int = 0,
    ):
        super().__init__()

        self.has_cross_attention = True

        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.num_heads = in_channels // self.attention_head_dim

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=self.num_heads),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=self.num_heads),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=self.num_heads),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=self.num_heads),  # Adjust dimensions
	    CrossAttentionLayer(mid_channels, 128, mid_channels, num_heads=self.num_heads),  # Adjust dimensions
	    CrossAttentionLayer(out_channels, 128, out_channels, num_heads=self.num_heads),  # Adjust dimensions
	]
	
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        if attention_mask is None:
            # if encoder_hidden_states is defined: we are doing cross-attn, so we should use cross-attn mask.
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # when attention_mask is defined: we don't even check for encoder_attention_mask.
            # this is to maintain compatibility with UnCLIP, which uses 'attention_mask' param for cross-attn masks.
            # TODO: UnCLIP should express cross-attn mask via encoder_attention_mask param instead of via attention_mask.
            #       then we can simplify this whole if/else block to:
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask
        
        hidden_states = self.down(hidden_states)

        hidden_states = self.resnets[0](hidden_states)
        
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # attn          
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            
            # Ensure encoder_hidden_states has consistent shape
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)  # Fix permutation inconsistency

            hidden_states = hidden_states.permute(0, 2, 1)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            else:
                encoder_hidden_states = encoder_hidden_states
                
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)

            hidden_states = attn(hidden_states, encoder_hidden_states, encoder_hidden_states, mask)
            hidden_states = hidden_states.permute(0, 2, 1)
            

            hidden_states = resnet(hidden_states)
            
            
        hidden_states = self.up(hidden_states)

        return hidden_states

        
def get_mid_block(
    mid_block_type: str,
    num_layers: int,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    embed_dim: int,
    add_downsample: bool,
    num_heads: int = 4  # Default number of attention heads
) -> nn.Module:
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)
    elif mid_block_type == "CrossAttentionMidBlock1D":
        return UNetMidBlock1DSimpleCrossAttn(
            in_channels=in_channels,
            temb_channels=embed_dim,
            out_channels=out_channels,
            mid_channels=mid_channels)
    else:
    	raise ValueError(f"{mid_block_type} does not exist.")
    	
    	
class CondUNet1DModel(ModelMixin, ConfigMixin):
    r"""
    CondUNet1DModel is a 1D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.
    Conditioning happens by appending class embedding to the input of the first down block. 
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model is initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`False`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock1D", "DownBlock1DNoSkip", "AttnDownBlock1D")`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpBlock1D", "UpBlock1DNoSkip", "AttnUpBlock1D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(32, 32, 64)`): Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to "UNetMidBlock1D"): block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): optional output processing of UNet.
        act_fn (`str`, *optional*, defaults to None): optional activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): group norm member count in UNet blocks.
        layers_per_block (`int`, *optional*, defaults to 1): added number of layers in a UNet block.
        downsample_each_block (`int`, *optional*, defaults to False:
            experimental feature for using a UNet without upsampling.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D",
                                        "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D",
                                      "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        act_fn: str = "silu",
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0],
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0],
                                       flip_sin_to_cos=flip_sin_to_cos,
                                       downscale_freq_shift=freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps,
                                          block_out_channels[0])
            timestep_input_dim = block_out_channels[0]
        time_embed_dim = block_out_channels[0] * 4

        # This is no longer optional
        # if use_timestep_embedding:

        self.time_mlp = TimestepEmbedding(
            in_channels=timestep_input_dim,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
        )

        # class embedding
        if class_embed_type == "embedding" and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds,
                                                extra_in_channels)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(1, extra_in_channels)
            # timestep_input_dim = block_out_channels[0]
        # identity does nothing, the class embedding needs to have the correct
        # dimensions already
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim,
                                               extra_in_channels)
        elif class_embed_type == "raw":
            self.class_embedding = "raw"
            assert extra_in_channels == 1
        else:
            self.class_embedding = None
        self.time_embed_dim = time_embed_dim
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
              input_channel += extra_in_channels
            	
              #input_channel = (extra_in_channels + input_channel + 6)

            is_final_block = i == len(block_out_channels) - 1
            

            down_block = get_down_block(
                down_block_type="CrossAttnDownBlock1D",
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            
            self.down_blocks.append(down_block)
            

        # mid
        self.mid_block = get_mid_block(
    	    mid_block_type="CrossAttentionMidBlock1D",
    	    num_layers=layers_per_block,
    	    in_channels=block_out_channels[-1],
    	    mid_channels=block_out_channels[-1],
    	    out_channels=block_out_channels[-1],
    	    embed_dim=block_out_channels[0],
    	    add_downsample=downsample_each_block,
    	    num_heads=2  # Specify the number of attention heads as needed
	)

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (reversed_block_out_channels[i + 1]
                              if i < len(up_block_types) -
                              1 else final_upsample_channels)

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type="CrossAttnUpBlock1D",
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(
            block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )
        
        # Projection
        self.projection = nn.Linear(64, 128)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        prev_step: torch.FloatTensor = None,
        return_dict: bool = True,        
    ) -> Union[CondUNet1DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): `(batch_size, num_channels, sample_size)` noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.UNet1DOutput`] instead of a plain tuple.
        Returns:
            [`~models.unet_1d.UNet1DOutput`] or `tuple`: [`~models.unet_1d.UNet1DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        
        # initially the shape is BxCxL (batch, channels, len)
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timestep_embed = self.time_proj(timesteps)

        # Time step embedding adds an additional projection to the initial timestep embedding
        timestep_embed = self.time_mlp(timestep_embed)

        # Generate class embedding as extra channels for the input
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )
            if self.config.class_embed_type == "timestep":
                # Applies time embedding first, (now class labels are floats)
                class_emb = self.class_embedding(
                    class_labels.unsqueeze(1)).to(dtype=self.dtype)

            if self.config.class_embed_type == "embedding":
                # Embedding needs input to be of type long
                class_emb = self.class_embedding(
                    class_labels.to(dtype=torch.long))
            if self.config.class_embed_type == "raw":
                class_emb = class_labels.to(dtype=self.dtype).unsqueeze(1)
                
            class_emb = class_emb.view(sample.shape[0], class_emb.shape[1],
                                       1).expand(sample.shape[0],
                                                 class_emb.shape[1],
                                                 sample.shape[2])
            sample = torch.cat((sample, class_emb), dim=1)
        # 2. down
        previous_features = None
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(sample, timestep_embed, encoder_hidden_states=prev_step)
            down_block_res_samples += res_samples
            
        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed, prev_step)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample,
                                    res_hidden_states_tuple=res_samples,
                                    temb=timestep_embed,
                                    encoder_hidden_states=prev_step)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample, )

        return CondUNet1DOutput(sample=sample)

