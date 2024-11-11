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

# @03/23/2023
# Fixing DDPM pipeline so it receives also a class for class conditioning generation

from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_cond: int = None,
        image_cond: torch.Tensor = None,
        guidance_scale: int = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            class_cond (`int`, *optional*):
                The value of the class to condition the image generation. 
            image_cond (`torch.Tensor`, *optional*):
                Tensor that contains the latent of the image to condition the image generation. 
            guidance_scale (`int`, *optional*):
                Parameter to control the classifier-free guidance step. Used when image_cond not None. 
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        self.guidance_scale = guidance_scale

        # Define the shape of the final images
        if isinstance(self.unet.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        
        # Define the initial noise tensor
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # If there is a conditioning image, generate the embedding of the image and
        # apply or not classifier-free guidance. 
        if image_cond is not None:
            image_cond_embedding = image_cond.view(image_cond.shape[0], -1)

            if self.do_classifier_free_guidance:
                image_uncond_embedding = torch.zeros_like(image_cond_embedding)
                image_cond_embedding = torch.cat([image_uncond_embedding, image_cond_embedding])

        # Main loop for the image generation 
        for t in self.progress_bar(self.scheduler.timesteps):
            # Expand the latents if we are doing classifier-free guidance 
            if image_cond is not None:
                image = torch.cat([image] * 2) if self.do_classifier_free_guidance else image
            
            # 1. Predict noise model_output
            label = None if class_cond is None else class_cond

            if image_cond is None:
                model_output = self.unet(sample=image,
                                        timestep=t,
                                        class_labels=label).sample
            else:
                model_output = self.unet(sample=image,
                                        timestep=t, 
                                        class_labels=label,
                                        encoder_hidden_states=None,
                                        added_cond_kwargs={'image_embed': image_cond_embedding})
            
            if image_cond is not None and self.do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)

            # 2. Compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample


        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
