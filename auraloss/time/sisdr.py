import torch

class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module."""

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
