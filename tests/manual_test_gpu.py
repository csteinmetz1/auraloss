import torch
import auraloss

y_hat = torch.randn(2, 1, 131072)
y = torch.randn(2, 1, 131072)

loss_fn = auraloss.freq.MelSTFTLoss(44100)
loss_fn2 = auraloss.freq.MultiResolutionSTFTLoss()

# loss_fn.cuda()

y_hat = y_hat.cuda()
y = y.cuda()

loss = loss_fn2(y_hat, y)
loss = loss_fn(y_hat, y)
