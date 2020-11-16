import torch
import auraloss

input = torch.rand(8,2,44100)
target = torch.rand(8,2,44100)

loss = auraloss.freq.SumAndDifferenceSTFTLoss()

print(loss(input, target))