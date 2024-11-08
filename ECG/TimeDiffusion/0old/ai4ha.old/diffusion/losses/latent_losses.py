# Adapted/copied from the code of Taming Transformers model
# https://github.com/CompVis/taming-transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lpips import LPIPS
class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class KLloss(nn.Module):
    """_Loss for VAE with KL divergence_

        KL VAE losses including:

        - KL (coming from Autoencoder KL model)
        - Perceptual loss computed as L1

    """

    def __init__(self, logvar_init=0.0, kl_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, weights=None):

        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(
            weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        return loss


class KLLPIPSloss(nn.Module):
    """_Loss for VAE with KL divergence_

        KL VAE losses including:

        - LPIPS
        - KL (coming from Autoencoder KL model)
        - Perceptual loss computed as L1

    """

    def __init__(self, params, logvar_init=0.0):

        super().__init__()
        self.perceptual_loss = LPIPS(params['root']).eval()
        # Loss weights for reconstruction, LPIPS, KL and discriminator
        self.kl_weight = params['kl_w']
        self.perceptual_weight = params['perceptual_w']
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, weights=None):

        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(
            weighted_nll_loss) / weighted_nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        return loss


class VAELPIPSWDloss(nn.Module):
    """_Loss for VAE with KL divergence_

        KL VAE losses including:

        - LPIPS
        - KL (coming from Autoencoder KL model)
        - Reconstruction loss computed as L1
        - Discriminator loss (coming from PatchGAN fake logits)

    """

    def __init__(self, params, logvar_init=0.0):

        super().__init__()

        self.perceptual_loss = LPIPS(params['root']).eval()

        # Loss weights for LPIPS, KL and discriminator
        self.kl_weight = params['kl_w']
        self.perceptual_weight = params['perceptual_w']
        self.disc_factor = params['discriminator_f']
        self.discriminator_weight = params['discriminator_w']
        self.discriminator_iter_start = params['disc_start']

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors,  logits_fake,
                global_step, last_layer=None, weights=None):

        # Reconstruction loss + LPIPS
        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # KL loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(
            weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part for the generator
        g_loss = -torch.mean(logits_fake)

        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                # assert not self.training
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(0.0)

        if self.discriminator_iter_start > global_step:
            loss = weighted_nll_loss + self.kl_weight * \
                kl_loss + d_weight * self.disc_factor * g_loss
        else:
            loss = weighted_nll_loss + self.kl_weight * kl_loss

        return loss


class VQLPIPSWDloss(nn.Module):
    """_Loss for Vector Quantized VAE_

        Vector Quantized VAE losses including:

        - LPIPS
        - Codebook loss (coming from VQ-VAE model)
        - Reconstruction loss computed as L1
        - Discriminator loss (coming from PatchGAN fake logits)

    """

    def __init__(self, params):

        super().__init__()

        self.perceptual_loss = LPIPS(params['root']).eval()

        # Loss weights for LPIPS, KL and discriminator
        self.codebook_weight = params['codebook_w']
        self.perceptual_weight = params['perceptual_w']
        self.disc_factor = params['discriminator_f']
        self.discriminator_weight = params['discriminator_w']
        self.discriminator_iter_start = params['disc_start']

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """ _Adaptive weight normalizing reconstruction loss and discriminator loss_

        From Taming Transformers paper
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self,  inputs, reconstructions, codebook_loss, logits_fake,
                global_step, last_layer=None, weights=None):

        # Reconstruction loss + LPIPS
        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # NLL loss
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part for the generator
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(
                nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            # assert not self.training
            d_weight = torch.tensor(0.0)

        if self.discriminator_iter_start > global_step:
            loss = nll_loss + d_weight * self.disc_factor * \
                g_loss + self.codebook_weight * codebook_loss.mean()
        else:
            loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        return loss


class DiscriminatorLoss(nn.Module):
    """_Patch GAN loss_

    Computes the discriminator loss using Hinge or Vanilla GAN loss
    """

    def __init__(self, params, disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        # Loss weights for LPIPS, KL and discriminator
        self.disc_factor = params['discriminator_f']
        self.discriminator_weight = params['discriminator_w']
        self.discriminator_iter_start = params['disc_start']

        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def forward(self, logits_real, logits_fake, global_step):
        # now the GAN part for the discriminator
        if self.discriminator_iter_start <= global_step:
            d_loss = self.disc_factor * \
                self.disc_loss(logits_real, logits_fake)
        else:
            d_loss = 0. * self.disc_factor * \
                self.disc_loss(logits_real, logits_fake)
        return d_loss


# class LPIPSWithDiscriminator(nn.Module):
#     def __init__(self, params, logvar_init=0.0, disc_loss="hinge"):

#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla"]
#         self.perceptual_loss = LPIPS(params['root']).eval()

#         # Loss weights for LPIPS, KL and discriminator
#         self.kl_weight = params['kl_w']
#         self.perceptual_weight = params['perceptual_w']
#         self.disc_factor = params['discriminator_f']
#         self.discriminator_weight = params['discriminator_w']
#         self.discriminator_iter_start = params['disc_start']

#         # output log variance
#         self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
#         self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         if last_layer is not None:
#             nll_grads = torch.autograd.grad(
#                 nll_loss, last_layer, retain_graph=True)[0]
#             g_grads = torch.autograd.grad(
#                 g_loss, last_layer, retain_graph=True)[0]
#         else:
#             nll_grads = torch.autograd.grad(
#                 nll_loss, self.last_layer[0], retain_graph=True)[0]
#             g_grads = torch.autograd.grad(
#                 g_loss, self.last_layer[0], retain_graph=True)[0]

#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight

#     def forward(self, inputs, reconstructions, posteriors, logits_real, logits_fake,
#                 optimizer_idx, global_step, last_layer=None, weights=None):

#         # Reconstruction loss + LPIPS
#         rec_loss = torch.abs(inputs.contiguous() -
#                              reconstructions.contiguous())
#         if self.perceptual_weight > 0:
#             p_loss = self.perceptual_loss(
#                 inputs.contiguous(), reconstructions.contiguous())
#             rec_loss = rec_loss + self.perceptual_weight * p_loss

#         # KL loss
#         nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
#         weighted_nll_loss = nll_loss
#         if weights is not None:
#             weighted_nll_loss = weights*nll_loss
#         weighted_nll_loss = torch.sum(
#             weighted_nll_loss) / weighted_nll_loss.shape[0]
#         nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
#         kl_loss = posteriors.kl()
#         kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

#         # now the GAN part for the generator
#         if optimizer_idx == 0:
#             g_loss = -torch.mean(logits_fake)

#             if self.disc_factor > 0.0:
#                 try:
#                     d_weight = self.calculate_adaptive_weight(
#                         nll_loss, g_loss, last_layer=last_layer)
#                 except RuntimeError:
#                     # assert not self.training
#                     d_weight = torch.tensor(0.0)
#             else:
#                 d_weight = torch.tensor(0.0)

#             if self.discriminator_iter_start > global_step:
#                 loss = weighted_nll_loss + self.kl_weight * \
#                     kl_loss + d_weight * self.disc_factor * g_loss
#             else:
#                 loss = weighted_nll_loss + self.kl_weight * kl_loss

#             return loss

#         # now the GAN part for the discriminator
#         if optimizer_idx == 1:
#             if self.discriminator_iter_start <= global_step:
#                 d_loss = self.disc_factor * \
#                     self.disc_loss(logits_real, logits_fake)
#             else:
#                 d_loss = 0. * self.disc_factor * \
#                     self.disc_loss(logits_real, logits_fake)
#             return d_loss
