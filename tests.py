import torch
from auraloss.time.fir import FIRFilter

filt = FIRFilter("aw", ntaps=101, plot=True)

input = (torch.rand(8,2,16000) * 2) - 1
target = (torch.rand(8,2,16000) * 2) - 1

input, target = filt(input, target)

print(input.shape)