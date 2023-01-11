import os
import torch
import auraloss


def test_mrstft():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)

    loss = auraloss.freq.MultiResolutionSTFTLoss()
    res = loss(pred, target)
    assert res is not None


def test_stft():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)

    loss = auraloss.freq.STFTLoss()
    res = loss(pred, target)
    assert res is not None


def test_stft_weights_a():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    # test difference weights
    loss = auraloss.freq.STFTLoss(
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_sc=1.0,
        reduction="mean",
    )
    res = loss(pred, target)
    assert res is not None


def test_stft_reduction():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    # test the reduction
    loss = auraloss.freq.STFTLoss(
        w_log_mag=1.0,
        w_lin_mag=1.0,
        w_sc=0.0,
        reduction="none",
    )
    res = loss(pred, target)
    print(res.shape)
    assert len(res.shape) > 1


def test_sum_and_difference():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    loss = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[512, 2048, 8192],
        hop_sizes=[128, 512, 2048],
        win_lengths=[512, 2048, 8192],
    )
    res = loss(pred, target)
    assert res is not None


def test_perceptual_sum_and_difference():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[512, 2048, 8192],
        hop_sizes=[128, 512, 2048],
        win_lengths=[512, 2048, 8192],
        perceptual_weighting=True,
        sample_rate=44100,
    )

    res = loss_fn(pred, target)
    assert res is not None


def test_melstft():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    # test MelSTFT
    loss = auraloss.freq.MelSTFTLoss(44100)
    res = loss(pred, target)
    assert res is not None


def test_melstft_reduction():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    # test MelSTFT with no reduction
    loss = auraloss.freq.MelSTFTLoss(44100, reduction="none")
    res = loss(pred, target)
    assert len(res) > 1


def test_multires_mel():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    sample_rate = 44100
    loss = auraloss.freq.MultiResolutionSTFTLoss(
        scale="mel",
        n_bins=64,
        sample_rate=sample_rate,
    )
    res = loss(pred, target)
    assert res is not None
