import logging
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
import torch
from diffusers.models import TransformerTemporalModel

from ai4ha.util import fix_paths, experiment_name_diffusion
from ai4ha.diffusion.pipelines.pipeline_ddpm_1d import DDPMPipeline
from ai4ha.diffusion.pipelines.pipeline_ddim import DDIMPipeline
from ai4ha.diffusion.models.unets.unet_1d import UNet1DModel
from ai4ha.diffusion.models.unets.cond_unet_1d import CondUNet1DModel
from ai4ha.diffusion.models.Transfusion import TransEncoder
from ai4ha.util.train import get_diffuser_scheduler
from ai4ha.util.sampling import sampling_diffusion

PIPELINES = {
    'DDPM': DDPMPipeline,
    'DDIM': DDIMPipeline
}

DIRS = ['gsamples']


def main(config):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    if 'class_embed_type' in config['model']['params']:
        class_conditioned = config['model']['params'][
            'class_embed_type'] is not None
    else:
        class_conditioned = False

    if 'num_class_embeds' in config['model']['params']:
        class_conditioned = config['model']['params'][
            'num_class_embeds'] is not None
    else:
        class_conditioned = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if config['model']['modeltype'] == "TTRANSFORMER":
        model = TransformerTemporalModel(**config['model']['params'])
    elif config['model']['modeltype'] == "UNET1C":
        model = CondUNet1DModel(**config['model']['params'])
    elif config['model']['modeltype'] == "UNET1":
        model = UNet1DModel(**config['model']['params'])
    elif config['model']['modeltype'] == "Transfusion":
        model = TransEncoder(**config['model']['params'])
    else:
        raise ValueError(
            f"Unsupported model type: {config['model']['modeltype']}")

    if class_conditioned and "nclasses" not in config['dataset']:
        raise ValueError(
            "Class conditioning is enabled but the number of classes is not specified"
        )
    # model.disable_gradient_checkpointing()

    # Initialize the scheduler
    noise_scheduler = get_diffuser_scheduler(config)

    # Prepare everything with our `accelerator`.
    diffuser = model.from_pretrained(f"{BASE_DIR}/model/unet")
    diffuser.eval()
    diffuser.to("cuda")
    generator = torch.Generator(device='cuda')

    pipeline = PIPELINES[config['diffuser']['type']](
        unet=diffuser,
        scheduler=noise_scheduler,
    )
    nsamp = (config['samples']['samples_gen'] // config['samples'][
        'sample_batch_size']) + 1

    sampling_diffusion(config, BASE_DIR, pipeline, generator,
                       class_conditioned, nsamp, 0, False, True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def timeDiffusionTrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])

    cfg['name'] = experiment_name_diffusion(cfg)
    print(cfg['name'])
    main(cfg)


if __name__ == "__main__":
    timeDiffusionTrain()
