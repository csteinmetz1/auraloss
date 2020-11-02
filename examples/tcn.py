import torch
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from auraloss.freq import MultiResolutionSTFTLoss, RandomResolutionSTFTLoss

def center_crop(x, shape):
    start = (x.shape[-1]-shape[-1])//2
    stop  = start + shape[-1]

    return x[...,start:stop]

class TCNBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=0, dilation=1, depthwise=False):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.depthwise = depthwise

        groups = out_ch if depthwise and (in_ch % out_ch == 0) else 1

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=padding, 
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        self.conv2 = torch.nn.Conv1d(out_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=padding, 
                                     dilation=1,
                                     groups=groups,
                                     bias=False)

        if depthwise:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)
            self.conv2b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        self.bn1 = torch.nn.BatchNorm1d(in_ch)
        self.bn2 = torch.nn.BatchNorm1d(out_ch)

        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.LeakyReLU()

        self.res = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):

        x_in = x

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        if self.depthwise:
            x = self.conv1b(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        if self.depthwise:
            x = self.conv2b(x)

        x_res = self.res(x_in)
        x = x + center_crop(x_res, x.shape)

        return x

class TCNModel(pl.LightningModule):
    """ Temporal convolutional network module.

        Params:
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            ninputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            depthwise (bool): Use depthwise-separable convolutions to reduce the total number of parameters. Default: False
        """
    def __init__(self, 
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=3, 
                 dilation_growth=1, 
                 channel_growth=1, 
                 channel_width=64, 
                 stack_size=10,
                 depthwise=False,
                 **kwargs):
        super(TCNModel, self).__init__()

        self.save_hyperparameters()

        self.rrstft = RandomResolutionSTFTLoss()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width

            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=kernel_size, 
                                        dilation=dilation,
                                        depthwise=depthwise))

        self.blocks.append(torch.nn.Sequential(
            torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)
        ))


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
            rf = rf + ((self.hparams.kernel_size-1) * 1)
        return rf

    def training_step(self, batch, batch_idx):
        s1, s2, noise, _ = batch

        # select either speaker 1 or 2 at random
        clean_speech = s1 if np.random.rand() > 0.5 else s2
        
        # pass the noisy speech through the model
        noisy_speech = clean_speech + noise
        pred_speech = self(noisy_speech)

        # crop the clean speech 
        clean_speech = center_crop(clean_speech, pred_speech.shape)

        # compute the error using appropriate loss
        #loss = torch.nn.functional.l1_loss(pred_speech,clean_speech)
        loss = self.rrstft(pred_speech, clean_speech)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        s1, s2, noise, _ = batch
        clean_speech = s1 if np.random.rand() > 0.5 else s2

        # pass the noisy speech through the model
        noisy_speech = clean_speech + noise
        pred_speech = self(noisy_speech)

        # crop the clean speech 
        clean_speech = center_crop(clean_speech, pred_speech.shape)

        # compute the error using appropriate loss
        #loss = torch.nn.functional.l1_loss(pred_speech,clean_speech)
        loss = self.rrstft(pred_speech, clean_speech)

        self.log('val_loss', loss)

        # move tensors to cpu for logging
        outputs = {
            "clean_speech" : clean_speech.cpu().numpy(),
            "pred_speech"  : pred_speech.cpu().numpy(),
            "noisy_speech" : noisy_speech.cpu().numpy(),
        }

        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = res = {k: v for d in validation_step_outputs for k, v in d.items()} 
        
        c = outputs["clean_speech"][0].squeeze()
        p = outputs["pred_speech"][0].squeeze()
        n = outputs["noisy_speech"][0].squeeze()

        # log audio examples
        self.logger.experiment.add_audio("clean", c, self.global_step, sample_rate=self.hparams.sample_rate)
        self.logger.experiment.add_audio("pred",  p, self.global_step, sample_rate=self.hparams.sample_rate)
        self.logger.experiment.add_audio("noisy", n, self.global_step, sample_rate=self.hparams.sample_rate)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=1)
        parser.add_argument('--nblocks', type=int, default=10)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--dilation_growth', type=int, default=1)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=64)
        parser.add_argument('--stack_size', type=int, default=10)
        parser.add_argument('--depthwise', type=bool, default=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser

if __name__ == '__main__':
    model = TCNModel(1, 1, 10, 3, 3, 1, 64, 10, True)
    rf = model.compute_receptive_field()
    torchsummary.summary(model, (1,131072))
    print(f"{(rf / 44100) * 1000:0.2f} ms")
