import os
import glob
import json
import torch
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from tcn import TCNModel
from data import SignalTrainLA2ADataset

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='./data')
parser.add_argument('--preload', type=bool, default=False)
parser.add_argument('--sample_rate', type=int, default=44100)
parser.add_argument('--logdir', type=str, default='./')
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--eval_subset', type=str, default='val')
parser.add_argument('--eval_length', type=int, default=262144)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)

# add model specific args
parser = TCNModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# parse them args
args = parser.parse_args()

# setup the dataloaders
test_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                      subset=args.eval_subset,
                                      preload=args.preload,
                                      length=args.eval_length)

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               shuffle=args.shuffle,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

results = {}

for model in 
    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path='/path/to/pytorch_checkpoint.ckpt',
        hparams_file='/path/to/test_tube/experiment/version/hparams.yaml',
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer.from_argparse_args(args)

    # test (pass in the model)
    res = trainer.test(model)

    # store in dict
    results[model_label] = res

# save final metrics to disk
