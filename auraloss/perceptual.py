import torch 
import numpy as np
import scipy.signal

from .plotting import compare_filters

class SumAndDifference(torch.nn.Module):
    """ Sum and difference signal extraction module."""
    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Input sum signal.
            Tensor: Input difference signal.
            Tensor: Target sum signal.
            Tensor: Target difference signal.
        """    
        if not (input.size(1) == target.size(1) == 2):
            raise ValueError("Input and target must be stereo.") # inputs must be stereo 
        input_sum = self.sum(input)
        input_diff = self.diff(input)
        target_sum = self.sum(target)
        target_diff = self.diff(target)

        return input_sum, input_diff, target_sum, target_diff
    
    @staticmethod
    def sum(self, x):
        return x[:,0,:] + x[:,1,:]

    @staticmethod
    def diff(self, x):
        return x[:,0,:] - x[:,1,:]


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

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1,1,-1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1,1,-1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2*np.pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
            DENs = np.polymul([1, 4*np.pi * f4, (2*np.pi * f4)**2],
                              [1, 4*np.pi * f1, (2*np.pi * f1)**2])
            DENs = np.polymul(np.polymul(DENs, [1, 2*np.pi * f3]),
                                               [1, 2*np.pi * f2])

            # convert analog filter to digital filter 
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=ntaps, bias=False, padding=ntaps//2)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype('float32')).view(1,1,-1)

            if plot: compare_filters(b, a, taps, fs=fs)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        bs,c,s = input.size()

        for i in range(c):
            input[:,i:i+1,:] = self.fir(input[:,i:i+1,:])
            target[:,i:i+1,:] = self.fir(target[:,i:i+1,:])

        return input, target