import torch
from typing import Any, List


def replace_denormals(x: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    """Replace numbers close to zero to avoid NaNs in `angle`"""
    y = x.clone()
    y[torch.abs(x) < threshold] = threshold
    return y


def angle(x: torch.Tensor) -> torch.Tensor:
    """Calculates the angle of a complex or real tensor"""
    if torch.is_complex(x):
        x_real = x.real
        x_imag = x.imag
    else:
        x_real = x
        x_imag = torch.zeros_like(x_real)

    x_real = replace_denormals(x_real)
    x_imag = replace_denormals(x_imag)
    return torch.atan2(x_imag, x_real)


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_length: int,
    win_length: int,
    window: str,
    eps: float = 1e-12,
):
    """Perform STFT.
    Args:
        x (Tensor): Input signal tensor (batch size, seq_len).

    Returns:
        Tensor: x_mag, x_phs
            Magnitude and phase spectra (batch_size, fft_size // 2 + 1, frames).
    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_length,
        win_length,
        window,
        return_complex=True,
    )
    x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=eps))
    x_phs = angle(x_stft)  # with fix to avoid NaNs in phase loss
    return x_mag, x_phs


class STFT(torch.nn.Module):
    """STFT transform module.

    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).

    Args:
        fft_size (int): FFT size in samples
        hop_size (int): Hop size of the FFT in samples
        win_length (int): Length of the FFT analysis window
        window (str, optional): Window to apply before FFT, options include:
           ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8

    Returns:
        stft_tensor

    """

    def __init__(
        self,
        fft_size: int,
        hop_length: int,
        win_length: int,
        window: str = "hann_window",
        sample_rate: Any = None,
        scale: Any = None,
        n_bins: Any = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.eps = eps

        # setup mel filterbank
        if scale is not None:
            try:
                import librosa.filters
            except Exception as e:
                print(e)
                print("Try `pip install auraloss[all]`.")

            if self.scale == "mel":
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(
                    sr=sample_rate,
                    n_fft=fft_size,
                    n_mels=n_bins,
                )
            elif self.scale == "chroma":
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate,
                    n_fft=fft_size,
                    n_chroma=n_bins,
                )
            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )

            self.register_buffer("fb", torch.tensor(fb).unsqueeze(0))

    def forward(self, x: torch.Tensor):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = stft(x)

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb, x_mag)

        x_mag_lin = x_mag
        x_mag_log = torch.log(x_mag)

        return x_mag_lin, x_mag_log, x_phs


class MultiResolutionSTFT(torch.nn.Module):
    """Multi resolution STFT transform module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
    """

    def __init__(
        self,
        fft_sizes: List[int],
        hop_sizes: List[int],
        win_lengths: List[int],
        window: str = "hann_window",
        sample_rate: Any = None,
        scale: Any = None,
        n_bins: Any = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_transforms = torch.nn.ModuleList()

        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_transforms += [
                STFT(
                    fs,
                    ss,
                    wl,
                    window,
                    sample_rate,
                    scale,
                    n_bins,
                    eps,
                )
            ]

    def forward(self, x: torch.Tensor):
        outputs = []
        for stft_transform in self.stft_transforms:
            x_mag_lin, x_mag_log, x_phs = stft_transform(x)
            outputs.append((x_mag_lin, x_mag_log, x_phs))

        return outputs


class SumAndDifference(torch.nn.Module):
    """Sum and difference stereo STFT transform module.

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

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return
