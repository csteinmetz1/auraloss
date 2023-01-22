import torch
import auraloss
import torchaudio
from tqdm import tqdm

def center_crop(x, length: int):
    start = (x.shape[-1]-length)//2
    stop  = start + length
    return x[...,start:stop]

def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[...,start:stop]

class TCNBlock(torch.nn.Module):
    def __init__(self, 
                in_ch, 
                out_ch, 
                kernel_size=3, 
                padding="same", 
                dilation=1, 
                grouped=False, 
                causal=False,
                **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        if padding == "same":
            pad_value = (kernel_size - 1) + ((kernel_size - 1) * (dilation-1))
        elif padding in ["none", "valid"]:
            pad_value = 0

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=0, # testing a change in padding was pad_value//2
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x):
        x_in = x

        x = self.conv1(x)
        #x = self.bn(x)
        x = self.relu(x)

        x_res = self.res(x_in)
        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])

        return x

class TCNModel(torch.nn.Module):
    """ Temporal convolutional network.
        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
            causal (bool): Causal TCN configuration does not consider future input values. Default: False
        """
    def __init__(self, 
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=3, 
                 dilation_growth=1, 
                 channel_growth=1, 
                 channel_width=32, 
                 stack_size=10,
                 grouped=False,
                 causal=False,):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth 
            else:
                out_ch = channel_width

            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=kernel_size, 
                                        dilation=dilation,
                                        padding="same" if causal else "valid",
                                        causal=causal,
                                        grouped=grouped,))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x):
        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return torch.tanh(self.output(x))

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return 

if __name__ == "__main__":

    y, sr = torchaudio.load("../sounds/assets/drum_kit_clean.wav")
    y /= y.abs().max()
    #y = y.repeat(2, 1)
    print(y.shape)

    # created distorted copy
    #x = torch.tanh(y * 4.0) 
    x = y + (0.01 * torch.randn(y.shape))
    
    # move data to gpu
    x = x.cuda()
    y = y.cuda()

    x = x.view(1, 2, -1)
    y = y.view(1, 2, -1)

    # create simple network
    model = TCNModel(ninputs=2, noutputs=2, kernel_size=13, dilation_growth=2)
    model.cuda()

    # create loss function
    loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        perceptual_weighting=True,
        sample_rate=44100,
        mel_stft=True,
        n_mel_bins=128,
    )
    loss_fn.cuda()

    #loss_fn = auraloss.freq.MultiResolutionSTFTLoss(    fft_sizes=[1024, 2048, 8192],
    #    hop_sizes=[256, 512, 2048],
    #    win_lengths=[1024, 2048, 8192],)

    #loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # run optimization
    pbar = tqdm(range(1000))
    for iter_idx in pbar:

        optimizer.zero_grad()

        y_hat = model(x)

        y_crop = causal_crop(y, y_hat.shape[-1])

        loss = loss_fn(y_hat, y_crop)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():0.4f}")

    torchaudio.save("tests/simple_train_gpu_output.wav", y_hat.detach().cpu().view(2,-1), sr)
    torchaudio.save("tests/simple_train_gpu_input.wav", x.detach().cpu().view(2,-1), sr)
