"""
PNDM Pipeline which allows the class and prompt conditioning of the image generation using CFG. 
"""
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.loaders import LoraLoaderMixin


class PNDMPipeline(DiffusionPipeline, LoraLoaderMixin): 
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

        self.clip_model = CLIPTextModel.from_pretrained('/gpfs/projects/bsc70/MN4/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0', subfolder='text_encoder')
        self.tokenizer = AutoTokenizer.from_pretrained('/gpfs/projects/bsc70/MN4/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0/tokenizer')

        self.register_modules(unet=unet, scheduler=scheduler)

        self.new_labels = {0: 'background/unknown',
                  1: 'stroma',
                  2: 'healthy epithelium',
                  3: '3+3',
                  4: '3+4',
                  5: '4+3',
                  6: '4+4',
                  7: '3+5',
                  8: '5+3',
                  9: '4+5',
                  10: '5+4',
                  11: '5+5'}

        class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        class_labels = class_labels.reshape(-1, 1)
        # One-hot encoding
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(class_labels)

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1 

    def apply_cfg(pos, neg, weight, rescale=0.7): 
        # Apply regular classifier-free guidance.
        cfg = neg + weight * (pos - neg)
        # Calculate standard deviations.
        std_pos = pos.std([1,2,3], keepdim=True)
        std_cfg = cfg.std([1,2,3], keepdim=True)
        # Apply guidance rescale with fused operations.
        factor = std_pos / std_cfg
        factor = rescale * factor + (1 - rescale)
        return cfg * factor

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_cond: int = None,
        guidance_scale: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        trailing: bool = False
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            class_cond (`int`, *optional*):
                The value of the class to condition the image generation. 
            embedding_type (`str`, *optional*):
                Type of embedding to use for the class conditioning. 
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
        image = randn_tensor(image_shape, generator=generator, device=generator.device)

        # Set step values
        if trailing:
            self.scheduler.config.timestep_spacing = "trailing"
        self.scheduler.set_timesteps(num_inference_steps)

        if encoder_hidden_states is not None:
            encoder_hidden_states = torch.tensor(encoder_hidden_states, device='cuda').unsqueeze(1).float()

        # Main loop for the image generation 
        for t in self.progress_bar(self.scheduler.timesteps):
            t = t.unsqueeze(0).to('cuda')
            # 1. Predict noise model_output
            label = None if class_cond is None else class_cond

            if self.do_classifier_free_guidance:
                model_output_cond = self.unet(image.to('cuda'),
                                            t,
                                            class_labels=label.to('cuda')).sample
                model_output_uncond = self.unet(image.to('cuda'),
                                                t,
                                                class_labels=None).sample

                model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = self.unet(hidden_states=image.to('cuda'),
                                        timestep=t, 
                                        class_labels=label.to('cuda')).sample

            # 2. Compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample


        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
