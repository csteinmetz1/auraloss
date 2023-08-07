import math
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


def test_stft_frame_normalization():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    # test difference weights
    loss = auraloss.freq.STFTLoss(
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_sc=1.0,
        reduction="mean",
        frame_normalization=True,
    )
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


def test_perceptual_mel_sum_and_difference():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        perceptual_weighting=True,
        sample_rate=44100,
        scale="mel",
        n_bins=128,
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


def test_perceptual_multires_mel():
    target = torch.rand(8, 2, 44100)
    pred = torch.rand(8, 2, 44100)
    sample_rate = 44100
    loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=sample_rate,
        perceptual_weighting=True,
    )
    res = loss(pred, target)
    assert res is not None


def test_stft_l2():
    N = 32
    n = torch.arange(N)

    f = N / 4
    target = torch.cos(2 * math.pi * f * n / N)
    target = target[None, None, :]
    pred = torch.zeros_like(target)

    loss = auraloss.freq.STFTLoss(
        fft_size=N,
        hop_size=N + 1,  # eliminate padding artefacts by enforcing only one hop
        win_length=N,
        window="ones",  # eliminate windowing artefacts
        w_sc=0.0,
        w_log_mag=0.0,
        w_lin_mag=1.0,
        w_phs=0.0,
        mag_distance="L2",
    )
    res = loss(pred, target)

    # MSE of energy in concentrated in single DFT bin
    expected_loss = ((N // 2) ** 2) / (N // 2 + 1)

    torch.testing.assert_close(res, torch.tensor(expected_loss), rtol=1e-3, atol=1e-3)


def test_multires_l2():
    N = 32
    n = torch.arange(N)

    f = N / 4
    target = torch.cos(2 * math.pi * f * n / N)
    target = target[None, None, :]
    pred = torch.zeros_like(target)

    loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[N],
        hop_sizes=[N + 1],  # eliminate padding artefacts by enforcing only one hop
        win_lengths=[N],
        window="ones",  # eliminate windowing artefacts
        w_sc=0.0,
        w_log_mag=0.0,
        w_lin_mag=1.0,
        w_phs=0.0,
        mag_distance="L2",
    )
    res = loss(pred, target)

    expected_loss = ((N // 2) ** 2) / (N // 2 + 1)

    torch.testing.assert_close(res, torch.tensor(expected_loss), rtol=1e-3, atol=1e-3)
