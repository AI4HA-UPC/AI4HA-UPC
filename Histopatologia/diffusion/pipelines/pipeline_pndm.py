"""
PNDM Pipeline which allows the class and prompt conditioning of the image generation using CFG. 
"""
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn

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

    def __init__(self, unet, scheduler, clip_model=None, tokenizer=None):
        super().__init__()

        if clip_model is None:
            self.clip_model = CLIPTextModel.from_pretrained('/gpfs/projects/bsc70/MN4/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0', subfolder='text_encoder')
        else:
            self.clip_model = clip_model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('/gpfs/projects/bsc70/MN4/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0/tokenizer')
        else:
            self.tokenizer = tokenizer

        self.register_modules(unet=unet, scheduler=scheduler)

        if isinstance(unet, nn.DataParallel):
            self.unet = unet.module

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
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(class_labels)

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

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
        embedding_type: str = None,
        guidance_scale: int = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_hidden_states: Optional[torch.Tensor] = None
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
        self.scheduler.set_timesteps(num_inference_steps)

        # If prompt conditioning is enabled, generate the prompt embedding and perfrom classifier-free guidance
        if embedding_type == 'encoder':
            prompts = []
            for i in class_cond:
                if i > 2:
                    prompt = f"Generate an histopathological image of Prostate Cancer of Gleason score {self.new_labels[i.item()]}"
                    prompts.append(prompt)
                else:
                    prompt = f"Generate an histopathological image of Prostate Cancer of label {self.new_labels[i.item()]}"
                    prompts.append(prompt)
            inputs = self.tokenizer(prompts, padding='max_length', max_length=48, return_tensors="pt")
            prompt_embeds = self.clip_model(inputs.input_ids, attention_mask=None)
            prompt_embeds = prompt_embeds[0]
            if self.do_classifier_free_guidance:
                uncond_prompts = [""] * batch_size
                uncond_inputs = self.tokenizer(uncond_prompts, padding='max_length', max_length=48, return_tensors="pt")
                uncond_prompt_embeds = self.clip_model(uncond_inputs.input_ids, attention_mask=None)
                uncond_prompt_embeds = uncond_prompt_embeds[0]
                prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])

        elif embedding_type == 'onehot':
            label_onehot = self.encoder.transform(class_cond.reshape(-1, 1))
            label_onehot = torch.tensor(label_onehot, device='cuda').unsqueeze(1).float()
            if self.do_classifier_free_guidance:
                labels_uncond_onehot = torch.zeros_like(label_onehot, device='cuda')
                prompt_embeds = torch.cat([labels_uncond_onehot, label_onehot])
            else:
                prompt_embeds = label_onehot

        if encoder_hidden_states is not None:
            encoder_hidden_states = torch.tensor(encoder_hidden_states, device='cuda').unsqueeze(1).float()

        # Main loop for the image generation 
        for t in self.progress_bar(self.scheduler.timesteps):
            # Expand the latents if we are doing classifier-free guidance 
            if embedding_type is not None:
                latent_image_input = torch.cat([image] * 2) if self.do_classifier_free_guidance else image
            
            # 1. Predict noise model_output
            label = None if class_cond is None else class_cond

            # Case prompt conditioning
            if embedding_type is not None:
                """
                model_output = self.unet(latent_image_input.to('cuda'),
                    t,
                    class_labels=label.to('cuda'),
                    encoder_hidden_states= encoder_hidden_states if encoder_hidden_states is not None else prompt_embeds.to('cuda')
                    ).sample
                """
                embd_cond, embd_uncond = prompt_embeds.chunk(2)
                model_output_cond = self.unet(image.to('cuda'),
                    t.to('cuda'),
                    class_labels=label.to('cuda'),
                    encoder_hidden_states=embd_cond.to('cuda')
                    ).sample
                model_output_uncond = self.unet(image.to('cuda'),
                    t.to('cuda'),
                    class_labels=None,
                    encoder_hidden_states=embd_uncond.to('cuda')
                    ).sample
                
            # Case label conditioning
            else:
                if self.do_classifier_free_guidance:
                    model_output_cond = self.unet(image,
                                                t,
                                                class_labels=label.to('cuda')).sample
                    model_output_uncond = self.unet(image,
                                                    t,
                                                    class_labels=None).sample

                    model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)
                else:
                    model_output = self.unet(sample=image,
                                            timestep=t, 
                                            class_labels=label).sample
            
            if (embedding_type is not None) and self.do_classifier_free_guidance:
                #model_output_uncond, model_output_cond = model_output.chunk(2)
                #model_output = self.apply_cfg(model_output_cond, model_output_uncond, self.guidance_scale)
                model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)
                # Calculate standard deviations.
                std_pos = model_output_cond.std([1,2,3], keepdim=True)
                std_cfg = model_output.std([1,2,3], keepdim=True)
                # Apply guidance rescale with fused operations.
                factor = std_pos / std_cfg
                factor = 0.7 * factor + (1 - 0.7)
                model_output = model_output * factor

            # 2. Compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample


        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
