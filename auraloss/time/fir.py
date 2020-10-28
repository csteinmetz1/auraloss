import torch
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt

class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap.

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright and Välimäki, 2019](https://arxiv.org/abs/1911.08922). 

    A-weighting filter - "aw"
    Folded differentiator - "fd"
    First-order highpass - "hp"

    Note that the default coefficeint value of 0.85 is optimized for 
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1,1,-1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1,1,-1)
        elif filter_type == "aw":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=101, bias=False, padding=101//2)
            self.fir.weight.requires_grad = False

            # first we define transfer function with poles
            z = [0,0]
            p = [-2*np.pi*20.598997057568145,
                 -2*np.pi*20.598997057568145,
                 -2*np.pi*12194.21714799801,
                 -2*np.pi*12194.21714799801]
            k = 1

            # normalize to 0dB at 1 kHz
            b, a = scipy.signal.zpk2tf(z, p, k)
            k /= abs(scipy.signal.freqs(b, a, [2*np.pi*1000])[1][0])

            # compute the analog frequency response
            w, h = scipy.signal.freqs_zpk(z, p, k, worN=1024)
            #w, _ = scipy.signal.freqz_zpk(z, p, k, fs=fs)

            # only take values within Nyquist
            w = w[np.where(w < fs//2)[0]]
            h = h[:len(w)]

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(101, w, abs(h), fs=fs)
            print(taps)
            self.fir.weight.data = torch.tensor(taps.astype('float32')).view(1,1,-1)

            # double check our filter
            w, h = scipy.signal.freqz(taps, fs=fs) 

            h_db = 20 * np.log10(abs(h))
            plt.plot(w, h_db)
            plt.xscale('log')
            plt.ylim([-50, 10])
            plt.xlim([10, 22.05e3])
            plt.grid()
            plt.show()

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

if __name__ == '__main__':

    filt = FIRFilter("aw")

    input = (torch.rand(8,2,16000) * 2) - 1
    target = (torch.rand(8,2,16000) * 2) - 1

    input, target = filt(input, target)

    print(input.shape)