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
from diffusers.models.unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block
import torch.nn.functional as F


@dataclass
class CondUNet1DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor

"""class CrossAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.projection_query = nn.Linear(in_features, out_features)
        self.projection_key = nn.Linear(in_features, out_features)
        self.projection_value = nn.Linear(in_features, out_features)
        self.multihead_attn = nn.MultiheadAttention(out_features, num_heads)

    def forward(self, query, key, value):
        # Projecting inputs to the same dimension
        query = self.projection_query(query.permute(2, 0, 1))
        key = self.projection_key(key.permute(2, 0, 1))
        value = self.projection_value(value.permute(2, 0, 1))
        # MultiheadAttention expects input as (L, N, E)
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output.permute(1, 2, 0)  # Reshape back to (N, E, L) if necessary
        
        
def interpolate_sequence(input_sequence, target_length):
    current_length = input_sequence.size(2)
    print(f'Interpolate current len: {current_length}')
    print(f'Target: {target_length}')
    if current_length == target_length:
        return input_sequence
    # Use linear interpolation for sequences
    #print(f'Interpolate input sequence: {input_sequence.shape}')
    #print(f'Interpolate input sequence unsqueeze: {input_sequence.unsqueeze(1).shape}')
    #print(f'Interpolate size: {target_length}')
    #print(f"Interpolate result: {F.interpolate(input_sequence, size=target_length, mode='linear', align_corners=False).squeeze(1).shape}")
    return F.interpolate(input_sequence, size=target_length, mode='linear', align_corners=False).squeeze(1)
    
    
		
class ModifiedDownBlock(nn.Module):
    def __init__(self, down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample, num_heads=1):
        super().__init__()
        self.down_block = get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample)
        self.cross_attention = CrossAttentionLayer(in_channels, out_channels, num_heads)

    def forward(self, x, temb, previous_features=None):
        print(f'Down initial x: {x.shape}')
        print(f'Down initial temb: {temb.shape}')
        x, res_samples = self.down_block(x, temb)
        print(f'Down then x: {x.shape}')
        print(f'Down then temb: {res_samples[0].shape}')
        if previous_features is not None:
            # Shape adjustment may be required depending on the actual shape of `x` and `previous_features`
            print(f'Down x: {x.shape}')
            print(f'Down previous_features: {previous_features.shape}')
            previous_features = interpolate_sequence(previous_features, x.size(2))
            x = self.cross_attention(x, previous_features, x.size(2))
        return x, res_samples
       
        

class ModifiedUpBlock(nn.Module):
    def __init__(self, up_block_type, num_layers, in_channels, out_channels, temb_channels, add_upsample, num_heads=1):
        super().__init__()
        self.up_block = get_up_block(up_block_type, num_layers, in_channels, out_channels, temb_channels, add_upsample)
        self.cross_attention = CrossAttentionLayer(in_channels, out_channels, num_heads)

    def forward(self, x, temb, res_hidden_states_tuple):
    	#print(f'res_hidden_states_tuple: {res_hidden_states_tuple[0].shape}')
    	#print(f'temb: {temb.shape}')
    	x = self.up_block(x, res_hidden_states_tuple, temb)
    	#print(f'Modified up enabled: {x.shape}')
    	# Assuming `res_hidden_states_tuple` contains features from corresponding down block
    	if res_hidden_states_tuple is not None:
    	    res_features = res_hidden_states_tuple[0]
    	    print(f'Up x = {x.shape}')
    	    print(f'Up res_features = {res_features.shape}')
    	    x = self.cross_attention(x.permute(2, 0, 1), res_features.permute(2, 0, 1), res_features.permute(2, 0, 1))
    	    x = x.permute(1, 2, 0)  # Reshape back if needed
    	return x"""

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
              # input_channel = (extra_in_channels + input_channel + 6)

            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            """down_block = ModifiedDownBlock(down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,)"""
            self.down_blocks.append(down_block)
            

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
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
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            """up_block = ModifiedUpBlock(up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,)"""
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
        
        #print(f"Input shape: {sample.shape}, Time step: {timestep}, Label: {class_labels}")
        
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
                                                 
            #print(f'DDIM: {sample.shape}, {class_emb.shape}')   
            #print(f'Initial sample: {sample.shape}')
            #print(f'Class emb: {class_emb.shape}')                                 
            sample = torch.cat((sample, class_emb), dim=1)
            # sample = torch.cat((sample, class_emb, prev_step), dim=1)
        # 2. down
        previous_features = None
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample,
                                                   temb=timestep_embed)
                                                   
            """print(f'Down sample: {sample.shape}')
            sample, res_samples = downsample_block(sample,
                                                   timestep_embed, previous_features=previous_features)"""
            down_block_res_samples += res_samples
            previous_features = sample
            #print(f'Projection: {previous_features.shape}')
            #previous_features = self.projection(previous_features.permute(0, 2, 1)).permute(0, 2, 1)
            
        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            
            #print(f'Res samples 0: {res_samples[0].shape}, {timestep_embed.shape}')
            #print(f'down_block_res_samples: {down_block_res_samples[0].shape}')
            """sample = upsample_block(sample,
                                    res_hidden_states_tuple=res_samples,
                                    temb=timestep_embed)"""
                                    
            #print(f'Sample: {sample.shape}')
            #print(f'Res_samples: {res_samples[0].shape}')
            #print(f'Timestep_embed: {timestep_embed.shape}')
            sample = upsample_block(sample,
                                    res_hidden_states_tuple=res_samples,
                                    temb=timestep_embed)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)

        if not return_dict:
            return (sample, )

        return CondUNet1DOutput(sample=sample)
