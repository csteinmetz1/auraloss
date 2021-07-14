import torch


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
