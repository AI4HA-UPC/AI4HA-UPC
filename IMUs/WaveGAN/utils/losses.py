import torch
from torch import autograd


def calc_gradient_penalty(net_dis, real_data, fake_data, labels=None):
    """ _Computes the gradient penalty for WGAN-GP_

    Copied from WGAN-GP torch implementation:

        https://github.com/KimRass/Conditional-WGAN-GP/blob/main/model.py
        
    Args:
        net_dis (torch.nn.Module): Discriminator network
        real_data (torch.Tensor): Real data
        fake_data (torch.Tensor): Fake data
    """
    # Compute interpolation factors
    alpha = torch.rand(real_data.size(0), 1, 1)
    alpha = alpha.expand(fake_data.size()).to(real_data.device)

    # Interpolate between real and fake data.
    target_size = fake_data.shape[2]
    tensor_size = real_data.shape[2]
    delta = tensor_size - target_size
    start = delta // 2
    end = tensor_size - (delta - start)
    real_data = real_data[:, :, start:end]
    
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(real_data.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates, labels=labels)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty

