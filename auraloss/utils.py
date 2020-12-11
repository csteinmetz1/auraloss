import torch

def apply_reduction(losses, reduction="none"):
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses