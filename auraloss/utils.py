import torch


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses

class FIRSequential(torch.nn.Sequential):
  def forward(self, *inputs):
    for module in self._modules.values():
      inputs = module(*inputs)
    return inputs