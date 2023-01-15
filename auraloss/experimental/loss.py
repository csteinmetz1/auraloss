import torch
from typing import Callable


class Loss(torch.nn.Module):
    def __init__(self, transform: torch.nn.Module, distance: Callable):
        self.transform = transform
        self.distance = distance
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        # apply transform to input and target
        input_t = self.transform(input)
        target_t = self.transform(target)

        # compute distance between input and target in transformed domain
        dist = self.distance(input_t, target_t)

        return x
