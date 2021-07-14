import torch
import numpy as np
import scipy.signal

import auraloss.plotting
import auraloss.utils
import auraloss.freq


class DelayInvariance(torch.nn.Module):
    """Compute the optimal time alignment between input and target."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def forward(self, input, target):
        """Calculate forward propagation.

        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            target (Tensor): Shifted groundtruth signal (B, #channels, #samples).
        """

        # downmix channels to mono
        mono_input = input.mean(dim=1)
        mono_target = target.mean(dim=1)

        # find the FFT size
        n_fft = auraloss.utils.next_power_of_2(mono_input.size(-1))

        X = torch.fft.rfft(mono_input, n=n_fft)
        Y = torch.fft.rfft(mono_target, n=n_fft)

        numerator = X * torch.conj(Y)
        denomenator = torch.abs(numerator)
        gcc = numerator / denomenator

        # find the estimated time delay
        tau = torch.argmax(gcc.abs())


class PerceptualMelSTFTLoss(torch.nn.Module):
    def __init__(self, sample_rate, **kwargs):
        super().__init__()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            sample_rate=sample_rate,
            scale="mel",
            n_bins=64,
            scale_invariance=True,
            **kwargs,
        )
        self.a_weighting = FIRFilter("aw", fs=sample_rate)

    def forward(self, input, target):

        # apply FIR A-weighting filter
        input, target = self.a_weighting(input.clone(), target.clone())

        # zero mean
        input_mean = torch.mean(input, dim=-1, keepdim=True)
        target_mean = torch.mean(target, dim=-1, keepdim=True)
        input = input - input_mean
        target = target - target_mean

        mrstft_loss = self.mrstft(input, target)

        return mrstft_loss


class SumAndDifference(torch.nn.Module):
    """Sum and difference signal extraction module."""

    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, #channels, #samples).
        Returns:
            Tensor: Sum signal.
            Tensor: Difference signal.
        """
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")

        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)

        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (list, optional): List of FFT sizes.
        hop_sizes (list, optional): List of hop sizes.
        win_lengths (list, optional): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'

    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'.
        loss, sum_loss, diff_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sum=1.0,
        w_diff=1.0,
        output="loss",
    ):
        super(SumAndDifferenceSTFTLoss, self).__init__()
        self.sd = SumAndDifference()
        self.w_sum = 1.0
        self.w_diff = 1.0
        self.output = output
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes, hop_sizes, win_lengths, window
        )

    def forward(self, input, target):
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                auraloss.plotting.compare_filters(b, a, taps, fs=fs)

    def forward(self, input, target):
        """Calculate forward propagation.

        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            input (Tensor): Filted predicted signal (B, #channels, #samples).
            target (Tensor): Filted groundtruth signal (B, #channels, #samples).
        """

        assert (
            input.shape == target.shape
        )  # check the shapes match for input and target

        bs, c, s = input.shape

        # since same filter is applied to all channels move channels onto batch dim
        input = input.view(bs * c, 1, -1)
        target = target.view(bs * c, 1, -1)

        input = torch.nn.functional.conv1d(
            input, self.fir.weight.data, padding=self.ntaps // 2
        )
        target = torch.nn.functional.conv1d(
            target, self.fir.weight.data, padding=self.ntaps // 2
        )

        # move channels back to channel dim
        input = input.view(bs, c, -1)
        target = target.view(bs, c, -1)

        return input, target
