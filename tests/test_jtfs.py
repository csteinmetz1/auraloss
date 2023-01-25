import os
import torch
import auraloss


def test_default_jtfs():
    bs = 2
    chs = 1
    seq_len = 32768
    loss_fn = auraloss.freq.TimeFrequencyScatteringLoss(seq_len)

    target = torch.rand(bs, chs, seq_len)
    pred = torch.rand(bs, chs, seq_len)

    res = loss_fn(pred, target)
    assert res is not None


def test_mae_jtfs():
    bs = 2
    chs = 1
    seq_len = 32768
    loss_fn = auraloss.freq.TimeFrequencyScatteringLoss(seq_len, dist=torch.nn.L1Loss)

    target = torch.rand(bs, chs, seq_len)
    pred = torch.rand(bs, chs, seq_len)

    res = loss_fn(pred, target)
    assert res is not None
