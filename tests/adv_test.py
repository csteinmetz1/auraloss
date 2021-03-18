import torch
import auraloss

input = torch.rand(8,2,44100)
target = torch.rand(8,2,44100)

sample_rate = 44100

mrmelstft_loss = auraloss.freq.MultiResolutionSTFTLoss(scale="mel", 
                                                       n_bins=64,
                                                       sample_rate=sample_rate)

melstft_loss = auraloss.freq.MelSTFTLoss(sample_rate)

print(melstft_loss(input, target))
print(mrmelstft_loss(input, target))