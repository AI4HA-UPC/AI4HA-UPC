""" 
Yu et al, Debias the Training of Diffusion Models
https://arxiv.org/abs/2310.08442
---
Salimans et al, Progressive distillation for fast sampling of diffusion models
https://arxiv.org/abs/2202.00512
---
Hang et al, Efficient Diffusion Training via Min-SNR Weighting Strategy
https://arxiv.org/abs/2303.09556
---
Choi et al, Perception Prioritized Training of Diffusion Models
https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html
"""
import torch


def loss_weighting(pred_type, method, snr, timesteps):
    """_Returns a weighting for the diffusion loss_

    Args:
        pred_type (_type_): _description_
        method (_type_): _description_
    """
    if pred_type == "epsilon":
        if method == "Yu":
            return 1.0 / torch.sqrt(snr)
        elif method == "Hang":
            # The paper says 5.0 works best
            return torch.stack(
                [5.0 / snr, torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
    elif pred_type == "sample":
        # use SNR weighting from distillation paper
        if method == "Salimans":
            return snr
        elif method == "Hang":
            # The paper says 5.0 works best
            return torch.stack([snr, 5.0 * torch.ones_like(timesteps)],
                               dim=1).min(dim=1)[0]
        elif method == "Choi":
            return 1 / (1 + snr)
    return torch.ones_like(timesteps)
