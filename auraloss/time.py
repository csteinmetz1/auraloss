import torch

class ESRLoss(torch.nn.Module):
    """Log-cosh loss function module. 
    
    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).
    """
    def __init__(self):
        super(ESRLoss).__init__()

    def forward(self, input, target):
        return torch.mean((torch.abs(target-input)**2)/(torch.abs(target)**2))

        import torch


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss function module. 
    
    See [Chen et al., 2019](https://openreview.net/forum?id=rkglvsC9Ym).
    """
    def __init__(self):
        super(LogCoshLoss).__init__()

    def forward(self, input, target):
        return torch.mean(torch.log(torch.cosh(input - target + 1e-12)))


class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.
    
    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)
    """

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SISDRLoss, self).__init__()

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: SI-SDR loss value.
        """
        return None
