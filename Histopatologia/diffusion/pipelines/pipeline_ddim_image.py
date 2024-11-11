
"""
Pipeline for the conditional image generation task using the DDIM Scheduler.
"""

from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers import DDIMScheduler



class DDIMImagePipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        #scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image_cond_latent: torch.Tensor = None,
        strength: float = 0.8,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Define the size of the output image
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        # Set the guidance scale and the device
        self._guidance_scale = guidance_scale
        device = self._execution_device

        # Generate the embedding of the conditional mask 
        image_cond_embedding = image_cond_latent.view(image_cond_latent.shape[0], -1)

        # Prepare latent (random noise)
        image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Perform classifier-free guidance 
        if self.do_classifier_free_guidance:
            image_uncond_embedding = torch.zeros_like(image_cond_embedding)
            image_cond_embedding = torch.cat([image_uncond_embedding, image_cond_embedding])

        for t in self.progress_bar(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([image] * 2) if self.do_classifier_free_guidance else image
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 1. predict noise model_output
            model_output = self.unet(latent_model_input, t,
                                          encoder_hidden_states=None, #image_cond_embedding.unsqueeze(1),
                                          added_cond_kwargs={'image_embeds': image_cond_embedding}
                                          ).sample
            
            if self.do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + self._guidance_scale * (model_output_cond - model_output_uncond)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)