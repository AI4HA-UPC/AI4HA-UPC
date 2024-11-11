"""
Pipeline for the conditional image generation task using the DDIM Scheduler and the concatenitaion method.
"""

from typing import List, Optional, Tuple, Union
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers import DPMSolverMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel



class DPMSolverMask(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        #scheduler = DDIMScheduler.from_config(scheduler.config)

        self.clip_model = CLIPTextModel.from_pretrained('/gpfs/projects/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0', subfolder='text_encoder')
        self.tokenizer = AutoTokenizer.from_pretrained('/gpfs/projects/bsc70/bsc70174/Models/stable-diffusion-xl-base-1.0/tokenizer')

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
    
    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        class_cond: int = None,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
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
        self.guidance_scale = guidance_scale
        device = self._execution_device

        # Generate the embedding of the empty prompt
        prompts = [f"Generate a pair of image and segmentation mask with a Gleason score of {self.new_labels[label.item()]}" for label in class_cond]
        inputs = self.tokenizer(prompts, padding='max_length', max_length=24, return_tensors="pt")
        prompt_embeds = self.clip_model(inputs.input_ids, attention_mask=None)
        prompt_embeds = prompt_embeds[0]

        # Prepare latent (random noise)
        image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Perform classifier-free guidance 
        if self.do_classifier_free_guidance:
            uncond_prompts = [""] * batch_size
            uncond_inputs = self.tokenizer(uncond_prompts, padding='max_length', max_length=24, return_tensors="pt")
            uncond_prompt_embeds = self.clip_model(uncond_inputs.input_ids, attention_mask=None)
            uncond_prompt_embeds = uncond_prompt_embeds[0]
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])

        for t in self.progress_bar(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_image = torch.cat([image] * 2) if self.do_classifier_free_guidance else image

            # 1. predict noise model_output
            model_output = self.unet(latent_image.to('cuda'),
                                    t,
                                    class_labels=class_cond.to('cuda'),
                                    encoder_hidden_states=prompt_embeds.to('cuda'), 
                                    ).sample
            
            if self.do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)