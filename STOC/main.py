import os
import torch

from .trainer import Trainer
from .dataloader import get_dataloader
from .config import load_config

import warnings

warnings.filterwarnings("ignore")


def main(args):
    # load data
    (
        full_len,
        trn_len,
        dev_len,
        tst_len,
        full_loader,
        trn_loader,
        dev_loader,
        tst_loader,
    ) = get_dataloader(args)

    # Create a trainer object for training
    trainer = Trainer(
        args,
        full_len,
        trn_len,
        dev_len,
        tst_len,
        full_loader,
        trn_loader,
        dev_loader,
        tst_loader,
    )

    # Train
    # trainer output: best checkpoint, version(the order of executions)
    best_result, version = trainer.fit()

    # Test
    # load the checkpoint to be tested
    data_path = args.root_path + args.data_name
    model_path = f"./saved_model/{args.experiment_name}/{data_path[3:-4]}/version-{version}/checkpoints/epoch-{best_result}.pt"
    ckpt = torch.load(model_path)

    # compute anomaly score, detect anomalies, compute performance measures, and plot the result
    trainer.test(ckpt, version)


if __name__ == "__main__":
    args = load_config()
    main(args)
