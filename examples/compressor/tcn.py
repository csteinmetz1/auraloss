import os
import torch
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

import auraloss

def center_crop(x, shape):
    start = (x.shape[-1]-shape[-1])//2
    stop  = start + shape[-1]
    return x[...,start:stop]

class FiLM(torch.nn.Module):
    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0,2,1)
        b = b.permute(0,2,1)

        x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class TCNBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=0, dilation=1, depthwise=False, conditional=False, **kwargs):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.depthwise = depthwise
        self.conditional = conditional

        groups = out_ch if depthwise and (in_ch % out_ch == 0) else 1

        self.conv1 = torch.nn.Conv1d(in_ch, 
                                     out_ch, 
                                     kernel_size=kernel_size, 
                                     padding=padding, 
                                     dilation=dilation,
                                     groups=groups,
                                     bias=False)
        if depthwise:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(out_ch, 128)
        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(in_ch, 
                                   out_ch, 
                                   kernel_size=1,
                                   groups=in_ch,
                                   bias=False)

    def forward(self, x, p=None):
        x_in = x

        x = self.conv1(x)
        if self.depthwise: # apply pointwise conv
            x = self.conv1b(x)
        if p is not None:   # apply FiLM conditioning
            x = self.film(x, p)
        else:
            x = self.bn(x)
        x = self.relu(x)

        x_res = self.res(x_in)
        x = x + center_crop(x_res, x.shape)

        return x

class TCNModel(pl.LightningModule):
    """ Temporal convolutional network with conditioning module.

        Params:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            depthwise (bool): Use depthwise-separable convolutions to reduce the total number of parameters. Default: False
            num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=3, 
                 dilation_growth=1, 
                 channel_growth=1, 
                 channel_width=64, 
                 stack_size=10,
                 depthwise=False,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(TCNModel, self).__init__()

        self.save_hyperparameters()

        # setup loss functions
        self.l1      = torch.nn.L1Loss()
        self.esr     = auraloss.time.ESRLoss()
        self.dc      = auraloss.time.DCLoss()
        self.logcosh = auraloss.time.LogCoshLoss()
        self.sisdr   = auraloss.time.SISDRLoss()
        self.stft    = auraloss.freq.STFTLoss()
        self.mrstft  = auraloss.freq.MultiResolutionSTFTLoss()
        self.rrstft  = auraloss.freq.RandomResolutionSTFTLoss()

        if nparams > 0:
            self.gen = torch.nn.Sequential(
                torch.nn.Linear(nparams, 32),
                torch.nn.PReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.PReLU(),
                torch.nn.Linear(64, 128),
                torch.nn.PReLU()
            )

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width

            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=kernel_size, 
                                        dilation=dilation,
                                        depthwise=self.hparams.depthwise,
                                        conditional=True if nparams > 0 else False))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, p=None):
        # if parameters present, 
        # compute global conditioning
        if p is not None:
            cond = self.gen(p)
        else:
            cond = None

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            if idx == 0:
                skips = x
            else:
                skips = center_crop(skips, x.shape) + x

        return torch.tanh(self.output(x + skips))

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
            rf = rf + ((self.hparams.kernel_size-1) * 1)
        return rf

    def training_step(self, batch, batch_idx):
        input, target, params = batch

        # pass the input thrgouh the mode
        pred = self(input, params)

        # crop the target signal 
        target = center_crop(target, pred.shape)

        # compute the error using appropriate loss      
        if   self.hparams.train_loss == "l1":
            loss = self.l1(pred, target)
        elif self.hparams.train_loss == "esr+dc":
            loss = self.esr(pred, target) + self.dc(pred, target)
        elif self.hparams.train_loss == "logcosh":
            loss = self.logcosh(pred, target)
        elif self.hparams.train_loss == "sisdr":
            loss = self.sisdr(pred, target)
        elif self.hparams.train_loss == "stft":
            loss = torch.stack(self.stft(pred, target),dim=0).sum()
        elif self.hparams.train_loss == "mrstft":
            loss = torch.stack(self.mrstft(pred, target),dim=0).sum()
        elif self.hparams.train_loss == "rrstft":
            loss = torch.stack(self.rrstft(pred, target),dim=0).sum()
        else:
            raise NotImplementedError(f"Invalid loss fn: {self.hparams.train_loss}")

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input, target, params = batch

        # pass the input thrgouh the mode
        pred = self(input, params)

        # crop the input and target signals
        input_crop = center_crop(input, pred.shape)
        target_crop = center_crop(target, pred.shape)

        # compute the validation error using all losses
        l1_loss      = self.l1(pred, target_crop)
        esr_loss     = self.esr(pred, target_crop)
        dc_loss      = self.dc(pred, target_crop)
        logcosh_loss = self.logcosh(pred, target_crop)
        sisdr_loss   = self.sisdr(pred, target_crop)
        stft_loss    = torch.stack(self.stft(pred, target_crop),dim=0).sum()
        mrstft_loss  = torch.stack(self.mrstft(pred, target_crop),dim=0).sum()
        rrstft_loss  = torch.stack(self.rrstft(pred, target_crop),dim=0).sum()
        
        aggregate_loss = l1_loss + \
                         esr_loss + \
                         dc_loss + \
                         logcosh_loss + \
                         sisdr_loss + \
                         mrstft_loss + \
                         stft_loss + \
                         rrstft_loss

        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)
        self.log('val_loss/ESR', esr_loss)
        self.log('val_loss/DC', dc_loss)
        self.log('val_loss/LogCosh', logcosh_loss)
        self.log('val_loss/SI-SDR', sisdr_loss)
        self.log('val_loss/STFT', stft_loss)
        self.log('val_loss/MRSTFT', mrstft_loss)
        self.log('val_loss/RRSTFT', rrstft_loss)

        # move tensors to cpu for logging
        outputs = {
            "input" : input_crop.cpu().numpy(),
            "target": target_crop.cpu().numpy(),
            "pred"  : pred.cpu().numpy(),
            "params" : params.cpu().numpy()}

        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : [],
            "params" : []}

        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx,...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            prm = outputs["params"][rand_idx].squeeze()

            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"target/{idx}", 
                                             t, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"pred/{idx}",   
                                             p, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                torchaudio.save(os.path.join(self.hparams.save_dir, 
                                f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.sample_rate)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

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
        parser.add_argument('--depthwise', default=False, action='store_true')
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--train_loss', type=str, default="l1")
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser