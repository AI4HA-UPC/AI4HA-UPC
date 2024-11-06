# Description: Script to evaluate the quality of the generated images using FID, mFID, KID and IS metrics.
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode


from accelerate import Accelerator
from accelerate.logging import get_logger

import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FLD import FLD
from fld.metrics.PrecisionRecall import PrecisionRecall

from tqdm.auto import tqdm

from ai4ha.util import instantiate_from_config, fix_paths

logger = get_logger(__name__, log_level="INFO")


def normalize_image(image):
    return torch.clamp(((image - image.min()) / (image.max() - image.min())),
                       min=0,
                       max=1)


def main(config):
    accparams = config["accelerator"]
    accelerator = Accelerator(**accparams)

    if config['validation']['metric'] == 'fid':
        fid = FrechetInceptionDistance(
            feature=config["validation"]["n_features"],
            normalize=True).to("cuda")
    elif config['validation']['metric'] == 'mfid':
        fid = MemorizationInformedFrechetInceptionDistance(
            feature=config["validation"]["n_features"],
            normalize=True).to("cuda")
    elif config['validation']['metric'] == 'kid':
        fid = KernelInceptionDistance(
            feature=config["validation"]["n_features"],
            normalize=True).to("cuda")
    elif config['validation']['metric'] == 'is':
        fid = InceptionScore(feature=config["validation"]["n_features"],
                             normalize=True).to("cuda")
    elif config['validation']['metric'] in ['fld', 'PR', 'RC']:
        pass
    else:
        raise ValueError(f"Unknown metric {config['validation']['metric']}")

    if config['validation']['metric'] in ['fid', 'kid', 'mfid']:
        logger.info("Loading real data")
        real_data = instantiate_from_config(config["dataset"]['train'])
        generated_data = instantiate_from_config(config["generated"]['train'])
        real_dataloader = torch.utils.data.DataLoader(real_data,
                                                      **config["dataloader"])
        dloader = accelerator.prepare(real_dataloader)

        logger.info(f"Computing {config['validation']['metric']} real data")
        for batch in tqdm(dloader):
            fid.update(normalize_image(batch["image"]).permute(0, 3, 1, 2),
                       real=True)

        logger.info("Loading generated data")
        generated_dataloader = torch.utils.data.DataLoader(
            generated_data, **config["dataloader"])
        dloader = accelerator.prepare(generated_dataloader)

        logger.info(
            f"Computing {config['validation']['metric']} generated data")
        for batch in tqdm(dloader):
            fid.update(normalize_image(batch["image"]).permute(0, 3, 1, 2),
                       real=False)
        logger.info(f"Final {config['validation']['metric']}={fid.compute()}")
    elif config['validation']['metric'] == 'is':
        logger.info("Loading generated data")
        generated_data = instantiate_from_config(config["generated"]['train'])
        generated_dataloader = torch.utils.data.DataLoader(
            generated_data, **config["dataloader"])
        dloader = accelerator.prepare(generated_dataloader)
        logger.info("Computing IS generated data")
        for batch in tqdm(dloader):
            fid.update(normalize_image(batch["image"]).permute(0, 3, 1, 2))
        logger.info(f"Final {config['validation']['metric']}={fid.compute()}")
    elif config['validation']['metric'] in ['PR', 'RC']:
        logger.info("Loading real data")
        real_data = instantiate_from_config(config["dataset"]['train'])
        generated_data = instantiate_from_config(config["generated"]['train'])
        real_dataloader = torch.utils.data.DataLoader(real_data,
                                                      **config["dataloader"])
        # dloader = accelerator.prepare(real_dataloader)
        dloader = accelerator.prepare(real_data)
        feature_extractor = InceptionFeatureExtractor()
        logger.info("Computing features real data")
        real_features = feature_extractor.get_dataset_features(real_data)
        logger.info("Computing features generated data")
        generated_features = feature_extractor.get_dataset_features(
            generated_data)
        nn = 5 if 'n_neighbors' not in config['validation'] else config[
            'validation']['n_neighbors']
        if config['validation']['metric'] == 'PR':
            fld = PrecisionRecall(mode="Precision",
                                  num_neighbors=nn).compute_metric(
                                      real_features, None, generated_features)
        else:
            fld = PrecisionRecall(mode="Recall",
                                  num_neighbors=nn).compute_metric(
                                      real_features, None, generated_features)
        logger.info(f"{config['validation']['metric']}={fld}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def TestQuality(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    main(cfg)


if __name__ == "__main__":
    TestQuality()
