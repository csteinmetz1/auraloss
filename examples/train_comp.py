import os
import glob
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
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=32768)
parser.add_argument('--eval_length', type=int, default=32768)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0)

# add model specific args
parser = TCNModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# parse them args
args = parser.parse_args()

# setup the dataloaders
train_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                subset=args.train_subset,
                                preload=args.preload,
                                length=args.train_length)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=args.shuffle,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

val_dataset = SignalTrainLA2ADataset(args.root_dir, 
                                preload=args.preload,
                                subset=args.val_subset,
                                length=args.eval_length)

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False,
                                             batch_size=2,
                                             num_workers=args.num_workers)


past_logs = sorted(glob.glob(os.path.join("lightning_logs", "*")))
if len(past_logs) > 0:
    version = int(os.path.basename(past_logs[-1]).split("_")[-1]) + 1
else:
    version = 0

# the losses we will test
losses = ["l1", "logcosh", "esr+dc", "stft", "mrstft", "rrstft"]

for loss_fn in losses:

    print(f"training with {loss_fn}")
    # init logger
    logdir = os.path.join("lightning_logs", f"version_{version}", loss_fn)
    print(logdir)
    args.default_root_dir = logdir

    # init the trainer and model 
    trainer = pl.Trainer.from_argparse_args(args)
    print(trainer.default_root_dir)

    dict_args = vars(args)
    dict_args["nparams"] = 2
    dict_args["train_loss"] = loss_fn
    model = TCNModel(**dict_args)

    # train!
    trainer.fit(model, train_dataloader, val_dataloader)
