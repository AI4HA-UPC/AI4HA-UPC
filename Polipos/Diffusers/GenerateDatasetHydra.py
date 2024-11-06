################################################################################
# Script for generating a dataser from a latent diffusion model based on t     #
# SD VAEs using diffusers library and Hydra.                                   #
#                                                                              #
# The dataset is generated jointly with the localization mask, this is         #
# converted into a list of bounding boxes using a find contours function       #
# an adequate scaling file for the generated VAE latens of the model is        #
# required for generating the final image                                      #
################################################################################
import inspect
import logging
import numpy as np
import os
from time import time
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode


import torch


from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from ai4ha.util import time_management, fix_paths, experiment_name_diffusion
from ai4ha.diffusion.pipelines.pipeline_ddpm import DDPMPipeline
from ai4ha.diffusion.pipelines.pipeline_ddim import DDIMPipeline
from ai4ha.Autoencoders import AutoencoderKL
import torchvision.transforms as T
from torchvision.utils import make_grid
from skimage.measure import label, regionprops, find_contours
import albumentations
import cv2

SCHEDULERS = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}

PIPELINES = {"DDPM": DDPMPipeline, "DDIM": DDIMPipeline}


def mask_to_border(mask):
    """Convert a mask to border image"""
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


def convert_to_yolo(anns, img_height, img_width):
    """_converts from (X,Y, H, W) to YOLO bounding box_

    Args:
        anns (_type_): _description_
        img_height (_type_): _description_
        img_width (_type_): _description_

    Returns:
        _type_: _description_
    """
    bboxes = []
    for ann in anns:
        x1, y1, w, h = ann
        x2 = x1 + w
        y2 = y1 + h
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        bboxes.append([x_center, y_center, w, h])
    return bboxes


def mask_to_bbox(mask, scale, size):
    """_Extracts a bounding box from a mask and returns it in yolo format_

    Args:
        mask (_type_): _description_
        scale (_type_): _description_
        size (_type_): _description_

    Returns:
        _type_: _description_
    """
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2 - x1, y2 - y1])

    return convert_to_yolo(np.array(bboxes) * scale, size, size)


def denorm(x, xmean, xstd):
    return x * xstd[:, None, None] + xmean[:, None, None]


def get_diffuser_scheduler(config):
    scheduler = SCHEDULERS[config["diffuser"]["type"]]

    params = {
        "num_train_timesteps": config["diffuser"]["num_steps"],
        "beta_schedule": config["diffuser"]["beta_schedule"],
        "clip_sample": False,
    }

    if "prediction_type" in set(
        inspect.signature(scheduler.__init__).parameters.keys()
    ):
        params["prediction_type"] = config["diffuser"]["prediction_type"]
    if (
        "variance_type" in set(inspect.signature(scheduler.__init__).parameters.keys())
    ) and ("variance_type" in config["diffuser"]):
        params["variance_type"] = config["diffuser"]["variance_type"]
    if "betas" in config["diffuser"]:
        params["beta_start"] = config["diffuser"]["betas"][0]
        params["beta_end"] = config["diffuser"]["betas"][1]

    return scheduler(**params)


def main(config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    if "num_inference_steps" in config["samples"]:
        config["diffuser"]["num_inference_steps"] = config["samples"][
            "num_inference_steps"]

    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    SAVE_DIR = f"{BASE_DIR}/gsamples/{config['diffuser']['type']}-{config['diffuser']['num_inference_steps']}"

    os.makedirs(f"{SAVE_DIR}", exist_ok=True)

    for d in ["images", "masks", "labels"]:
        os.makedirs(f"{SAVE_DIR}/{d}", exist_ok=True)

    logger.info("LOADING VAE")
    aekl = AutoencoderKL.from_pretrained(config["latents"]["model"])
    aekl.eval()
    if config["latents"]["cuda"]:
        aekl.to("cuda")

    logger.info("LOADING LATENTS NORMALIZATION")
    dataset = config["dataset"]
    norm_path = f"{config['latents']['norm_path']}/{dataset['norm']}-{dataset['train']['params']['model']}-{dataset['train']['params']['size']}-{dataset['train']['params']['encoder']}"
    xmean = np.load(f"{norm_path}-mean.npy")
    xstd = np.load(f"{norm_path}-std.npy")

    logger.info("LOADING MODEL")
    diffuser = UNet2DModel.from_pretrained(f"{BASE_DIR}/model/unet")
    if "samples_latent_size" in config["samples"]:
        diffuser.config.sample_size = config["samples"]["samples_latent_size"]

    diffuser.eval()
    diffuser.to("cuda")

    logger.info("LOADING SCHEDULER and PIPELINE")
    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)
    if "pipeline" not in config["diffuser"]:
        pipeline = PIPELINES[config["diffuser"]["type"]](
            unet=diffuser,
            scheduler=noise_scheduler,
        )
    else:
        pipeline = PIPELINES[config["diffuser"]["pipeline"]](
            unet=diffuser,
            scheduler=noise_scheduler,
        )
    generator = torch.Generator(device="cuda")
    seed = config["samples"]["seed"]
    generator.manual_seed(seed)

    last_time = time()
    qe_time = []

    if "time" in config:
        time_budget = (config["time"]) * 60
    else:
        time_budget = 2000

    rescaler = albumentations.SmallestMaxSize(
        max_size=config["samples"]["samples_size"],
        interpolation=cv2.INTER_NEAREST)

    logger.info("GENERATING SAMPLES")
    for i in range((config["samples"]["gen_num_samples"]//config["diffuser"]["batch_size"])+1):
        images = pipeline(
            generator=generator,
            batch_size=config["diffuser"]["batch_size"],
            num_inference_steps=config["samples"]["num_inference_steps"],
            output_type="numpy",
            latent=True,
            vae=None,
        ).images

        images_chann = torch.from_numpy(images[..., :4])
        masks_chann = torch.from_numpy(images[..., -1])
        for img in range(images.shape[0]):
            # Generate image
            sample_d = denorm(images_chann[img].permute(2, 0, 1), xmean, xstd)
            if config["latents"]["cuda"]:
                sample_d = sample_d.to("cuda")
            sample_dec = (aekl.decode(torch.unsqueeze(
                sample_d, dim=0)).sample.detach().cpu().numpy().squeeze())
            image = torch.clamp(
                (torch.from_numpy((sample_dec + 1) / 2)),
                min=0,
                max=1,
            )
            grid = T.ToPILImage(mode="RGB")(make_grid(image, nrow=1))
            grid.save(
                f"{SAVE_DIR}/images/image_{seed:010d}_{i:05d}_{img:02d}.png",
                compress_level=6)

            # Generate mask
            mask = masks_chann[img]  # .permute(2, 0, 1)
            mask = rescaler(image=mask.numpy())["image"]
            if 'invert_mask' in config['samples'] and config['samples'][
                    'invert_mask']:
                mask = np.where(mask < 0.5, 0, 1)
            else:
                mask = np.where(mask > 0.5, 0, 1)
            grid = T.ToPILImage()(make_grid(
                torch.tensor(mask.astype(np.uint8)).unsqueeze(0) * 255,
                nrow=1))
            grid.save(
                f"{SAVE_DIR}/masks/image_{seed:010d}_{i:05d}_{img:02d}.png",
                compress_level=6)

            # Bounding boxes
            scale = sample_dec.shape[1] / masks_chann.shape[1]
            size = sample_dec.shape[1]
            rec = mask_to_bbox(masks_chann[img].numpy() * 256, scale, size)
            bbfile = open(
                f"{SAVE_DIR}/labels/image_{seed:010d}_{i:05d}_{img:02d}.txt",
                "w")
            for r in rec:
                bbfile.write(f"0 {r[0]} {r[1]} {r[2]} {r[3]}\n")

            bbfile.close()

        last_time, qe_time, time_budget = time_management(
            last_time, qe_time, time_budget, logger)
        if time_budget < np.mean(qe_time):
            break


@hydra.main(version_base=None, config_path="conf", config_name="config")
def GenerateDataset(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    cfg['name'] = experiment_name_diffusion(cfg)
    main(cfg)


if __name__ == "__main__":
    GenerateDataset()
