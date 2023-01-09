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
parser.add_argument("--root_dir", type=str, default="./data")
parser.add_argument("--preload", type=bool, default=False)
parser.add_argument("--sample_rate", type=int, default=44100)
parser.add_argument("--logdir", type=str, default="./")
parser.add_argument("--eval_subset", type=str, default="val")
parser.add_argument("--eval_length", type=int, default=262144)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=0)

# add model specific args
parser = TCNModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# parse them args
args = parser.parse_args()

# setup the dataloaders
test_dataset = SignalTrainLA2ADataset(
    args.root_dir,
    subset=args.eval_subset,
    preload=args.preload,
    length=args.eval_length,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

results = {}

# the losses we will test
losses = ["l1", "logcosh", "esr+dc", "stft", "mrstft", "rrstft"]

for loss_model in losses:

    root_logdir = os.path.join(args.logdir, loss_model, "lightning_logs", "version_0")

    checkpoint_path = glob.glob(os.path.join(root_logdir, "checkpoints", "*"))[0]
    print(checkpoint_path)
    hparams_file = os.path.join(root_logdir, "hparams.yaml")

    model = TCNModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location="cuda:0",
    )

    model.hparams.save_dir = args.save_dir
    model.hparams.num_examples = args.num_examples

    # init trainer with whatever options
    trainer = pl.Trainer.from_argparse_args(args)

    # set the seed
    pl.seed_everything(42)

    # test (pass in the model)
    res = trainer.test(model, test_dataloaders=test_dataloader)

    # store in dict
    results[loss_model] = res

# save final metrics to disk
with open(os.path.join("examples", "compressor", "comp-metrics.json"), "w") as fp:
    json.dump(results, fp, indent=True)
