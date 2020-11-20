import torch

class ESRLoss(torch.nn.Module):
    """Error-to-signal ratio loss function module. 
    
    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).
    """
    def __init__(self):
        super(ESRLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(target-input)**2)/torch.mean(torch.abs(target)**2)


class DCLoss(torch.nn.Module):
    """DC loss function module. 
    
    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).
    """
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, input, target):
        return (torch.abs(torch.mean(target-input))**2)/(torch.mean(torch.abs(target)**2))


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss function module. 
    
    See [Chen et al., 2019](https://openreview.net/forum?id=rkglvsC9Ym).
    """
    def __init__(self, a=1.0, eps=1e-12):
        """Initilize Log-cosh loss module
        Args:
            a (float): Smoothness hyperparameter. Smaller is smoother. Default: 1.0
            eps (float): Small epsilon value for stablity. Default: 1e-12
        """
        super(LogCoshLoss, self).__init__()
        self.a = a
        self.eps = eps

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Log cosh loss value.
        """
        return torch.mean( (1/self.a) * torch.log(torch.cosh(self.a * (input - target)) + self.eps))


class SDRLoss(torch.nn.Module):
    """Signal-to-distortion ratio loss module.

    See [Vincent et al., 2006](https://ieeexplore.ieee.org/document/1643671)
    """

    def __init__(self):
        """Initilize SDR loss module."""
        super(SDRLoss, self).__init__()
        raise NotImplementedError()

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: SDR loss value.
        """
        return None

class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Args:
        eps (float): Small epsilon value for stablity. Default: 1e-12
    
    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)
    """

    def __init__(self, eps=1e-12):
        """Initilize SI-SDR loss module."""
        super(SISDRLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: SI-SDR loss value.
        """
        bs,c,s = input.size()
        alpha = (input * target).sum(-1) / (target ** 2).sum(-1)
        res = input - (target * alpha.view(bs,c,1))

        return 10 * torch.log10(((target**2).sum()/(res**2).sum().clamp(self.eps)))

