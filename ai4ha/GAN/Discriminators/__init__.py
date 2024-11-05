# Path: ai4ha/GAN/Generators/__init__.py

from .Discriminators import NLayerDiscriminator, weights_init, Discriminator
from .WaveGANDiscriminator import WaveGANDiscriminator, wavegan_discriminator_exp_name
from .TTSGANDiscriminator import TTSGANDiscriminator, ttsgan_discriminator_exp_name

__all__ = [
    "NLayerDiscriminator", "WaveGANDiscriminator", "TTSGANDiscriminator", 
    "wavegan_discriminator_exp_name", "ttsgan_discriminator_exp_name", "weights_init",
    "Discriminator"
]
