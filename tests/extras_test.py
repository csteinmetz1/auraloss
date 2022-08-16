import torch
import auraloss

pred = torch.rand(8, 2, 44100)
target = torch.rand(8, 2, 44100)



# test MelSTFT
loss = auraloss.freq.MelSTFTLoss(44100)
res = loss(pred, target)
print(res, res.shape)
assert res is not None

# test MelSTFT with no reduction
loss = auraloss.freq.MelSTFTLoss(44100, reduction="none")
res = loss(pred, target)
print(res.shape)
assert len(res) > 1