import hydra
from omegaconf import DictConfig, OmegaConf, SCMode


import timm
from ai4ha.util import (instantiate_from_config, fix_paths)
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
from tqdm.auto import tqdm


def get_features_batch(model, transforms, batch):
    with torch.no_grad():
        return model(
            torch.cat([
                transforms(to_pil_image(image)).unsqueeze(0) for image in batch
            ]).to('cuda')).squeeze().cpu().numpy()


def main(config):
    model = timm.create_model(
        'vgg19.tv_in1k',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.to('cuda').eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    if 'dataset' in config:
        train_data = instantiate_from_config(config['dataset']['train'])
    elif 'generated' in config:
        train_data = instantiate_from_config(config['generated']['train'])

    config['dataloader']['shuffle'] = False
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   **config["dataloader"])

    lfeatures = []
    for batch in tqdm(train_dataloader):
        lfeatures.append(
            get_features_batch(model, transforms,
                               (batch['image'].permute(0, 3, 1, 2) + 1) *
                               127.5))

    vecimgg = np.concatenate(lfeatures)
    if 'dataset' in config:
        np.savez_compressed(config['dataset']['name'], embed=vecimgg)
    elif 'generated' in config:
        np.savez_compressed(config['generated']['name'], embed=vecimgg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def Embed(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)

    cfg = fix_paths(cfg, cfg['local'])
    main(cfg)


if __name__ == "__main__":
    Embed()
