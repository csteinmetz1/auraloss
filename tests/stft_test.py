import torch
import auraloss

target = torch.rand(8, 2, 44100)
pred = torch.rand(8, 2, 44100)

# test MR-STFT
loss = auraloss.freq.MultiResolutionSTFTLoss()
res = loss(pred, target)
print(res, res.shape)
assert res is not None

# test STFT
loss = auraloss.freq.STFTLoss()
res = loss(pred, target)
print(res, res.shape)
assert res is not None

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

# test difference weights
loss = auraloss.freq.STFTLoss(w_log_mag=1.0, w_lin_mag=0.0, w_sc=1.0, reduction="mean")
res = loss(pred, target)
print(res, res.shape)
assert res is not None

# test the reduction
loss = auraloss.freq.STFTLoss(w_log_mag=1.0, w_lin_mag=1.0, w_sc=0.0, reduction="none")
res = loss(pred, target)
print(res.shape)
assert len(res.shape) > 1
