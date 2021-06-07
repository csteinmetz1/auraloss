import torch
from .utils import apply_reduction


class ESRLoss(torch.nn.Module):
    """Error-to-signal ratio loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, reduction="mean"):
        super(ESRLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        losses = ((target - input).abs() ** 2).sum(-1) / (target.abs() ** 2).sum(-1)
        losses = apply_reduction(losses, reduction=self.reduction)
        return losses


class DCLoss(torch.nn.Module):
    """DC loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, reduction="mean"):
        super(DCLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        losses = ((target - input).mean(-1) ** 2).abs() / (target.abs().mean(-1) ** 2)
        losses = apply_reduction(losses, self.reduction)
        return losses


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss function module.

    See [Chen et al., 2019](https://openreview.net/forum?id=rkglvsC9Ym).

    Args:
        a (float, optional): Smoothness hyperparameter. Smaller is smoother. Default: 1.0
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, a=1.0, eps=1e-8, reduction="mean"):
        super(LogCoshLoss, self).__init__()
        self.a = a
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        losses = (
            (1 / self.a) * torch.log(torch.cosh(self.a * (input - target)) + self.eps)
        ).mean(-1)
        losses = apply_reduction(losses, self.reduction)
        return losses


class SNRLoss(torch.nn.Module):
    """Signal-to-noise ratio loss module.

    Note that this does NOT implement the SDR from
    [Vincent et al., 2006](https://ieeexplore.ieee.org/document/1643671),
    which includes the application of a 512-tap FIR filter.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SNRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        res = input - target
        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses


class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Note that this returns the negative of the SI-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses


class SDSDRLoss(torch.nn.Module):
    """Scale-dependent signal-to-distortion ratio loss module.

    Note that this returns the negative of the SD-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-8, reduction="mean"):
        super(SDSDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean

        alpha = (input * target).sum(-1) / (((target ** 2).sum(-1)) + self.eps)
        scaled_target = target * alpha.unsqueeze(-1)
        res = input - target

        losses = 10 * torch.log10(
            (scaled_target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps
        )
        losses = apply_reduction(losses, self.reduction)
        return -losses


class MultiScaleDynamicLoss(torch.nn.Module):
    """Mutli-scale dynamic time domain loss module.

    See [Tian, et al., 2020](https://arxiv.org/abs/2011.12206)

    "Multi-scale dynamic loss consists of three parts in four resolution scales:
    loss_e, loss_p and loss_t which are designed to capture energy, fast convergence
    and remove high-frequency metallic noise respectively."

    Args:
        frame_sizes (list): List of ints specifying the frame sizes of each scale in samples.
        hop_sizes (list): List of ints specifying the hope sizes of each scale in samples.
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(
        self,
        frame_sizes=[1, 240, 480, 960],
        hop_sizes=[1, 120, 240, 480],
        eps=1e-8,
        reduction="mean",
    ):
        super(MultiScaleDynamicLoss, self).__init__()
        self.eps = eps
        self.frame_sizes = frame_sizes
        self.hop_sizes = hop_sizes
        self.reduction = reduction

    def energy_loss(self, input, target):
        return torch.norm(target ** 2 - input ** 2)

    def time_loss(self, input, target):
        return torch.norm(target - input)

    def phase_loss(self, input, target):
        return torch.norm(torch.diff(target) - torch.diff(input))

    def forward(self, input, target):

        loss = 0

        for frame_size, hop_size in zip(self.frame_sizes, self.hop_sizes):
            input_frames = input.unfold(-1, frame_size, hop_size)
            target_frames = target.unfold(-1, frame_size, hop_size)

            loss_e = self.energy_loss(input_frames, target_frames)
            loss_t = self.time_loss(input_frames, target_frames)
            loss_p = self.phase_loss(input_frames, target_frames)

            loss += loss_e + loss_t + loss_p

        return loss
