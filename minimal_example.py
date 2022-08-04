from typing import List
from pathlib import Path

from torchinfo import summary

import argparse
import collections
from collections import OrderedDict

from brain_dataset import BrainDataset
from model import BMENet
import numpy as np
import pandas as pd
from reader import NiftiFixedVolumeReader
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from catalyst import metrics
from catalyst.callbacks import CheckpointCallback
#from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose
from catalyst.dl import Runner, DeviceEngine, DataParallelEngine
from catalyst.metrics.functional._segmentation import dice

#from sklearn.metrics import mean_absolute_percentage_error

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_loaders(
    worker_init_fn,
    random_state: int,
    volume_shape: List[int],
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    batch_size: int = 1,
    num_workers: int = 64,
) -> dict:
    """Get Dataloaders"""
    datasets = {}
    open_fn = NiftiFixedVolumeReader(input_key="volume_paths")

    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()

    for mode, source in zip(
        ("train", "validation", "infer"),
        (in_csv_train, in_csv_valid, in_csv_infer),
    ):
        if source is not None and len(source) > 0:
            dataset = BrainDataset(
                df=pd.read_csv(source),
                open_fn=open_fn,
                input_key="images",
            )

        datasets[mode] = {"dataset": dataset}

        loader = DataLoader(
            dataset=datasets[mode]["dataset"],
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        if mode in ('train', 'valid'):
            train_loaders[mode] = BatchPrefetchLoaderWrapper(loader)
        else:
            infer_loaders[mode] = BatchPrefetchLoaderWrapper(loader)

    return train_loaders, infer_loaders


class CustomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, parallel: bool):
        """Init."""
        super().__init__()
        self.parallel = parallel

    def get_engine(self):
        """Gets engine for multi or single gpu case"""
        if self.parallel:
            engine = DataParallelEngine()

        else:
            engine = DeviceEngine()

        return engine

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "normalizedLoss"]
        }

    def handle_batch(self, batch):
        """
        Custom train/ val step that includes batch unpacking, training, and
        DICE metrics
        """
        # model train/valid step

        x, y = batch[0], batch[1]

        if self.is_train_loader:
            self.optimizer.zero_grad()

        y_hat = self.model(x)

        loss = F.mse_loss(y_hat, y)
        zeros = torch.zeros_like(y)
        normalized_loss = F.mse_loss(y_hat, y) / F.mse_loss(y, zeros) 
        #mape = mean_absolute_percentage_error(y, y_hat) 

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        self.batch_metrics.update({"loss": loss})
        self.batch_metrics.update({"normalizedLoss": normalized_loss})

        for key in ["loss"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )
        
        for key in ["normalizedLoss"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 segmentation Training")

    parser.add_argument(
        "--train_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/bin/pythonFiles/AIBM/training/data/train.csv",
        help="Path to list with brains for training",
    )
    parser.add_argument(
        "--validation_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/bin/pythonFiles/AIBM/training/data/val.csv",
        help="Path to list with brains for validation",
    )
    parser.add_argument(
        "--inference_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/bin/pythonFiles/AIBM/training/data/test.csv",

        help="Path to list with brains for inference",
    )
    parser.add_argument("--n_classes", default=68, type=int)
    parser.add_argument("--model", default="bmenet")
    parser.add_argument("--parallel", default=False)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--n_epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--lr", default=0.0003, type=float)
    args = parser.parse_args()
    print("{}".format(args))

    volume_shape = [256, 256, 256]
    train_loaders, infer_loaders = get_loaders(
        worker_init_fn,
        0,
        volume_shape,
        args.train_path,
        args.validation_path,
        args.inference_path,
        num_workers=args.num_workers
    )

    if args.model == "bmenet":
        net = BMENet(
            n_classes=args.n_classes,
        )
    else:
        raise ValueError('only bmenet supported')

    logdir = "/data/users2/vjoshi6/bin/pythonFiles/AIBM/15mostproblematic{lr}".format(lr = args.lr)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.02,
        epochs=args.n_epochs,
        steps_per_epoch=len(train_loaders["train"]),
    )

    Path(logdir).mkdir(parents=True, exist_ok=True)

    runner = CustomRunner(parallel=args.parallel)
"""
    runner.train(
        model=net,
        optimizer=optimizer,
        loaders=train_loaders,
        num_epochs=args.n_epochs,
        scheduler=scheduler,
        callbacks=[CheckpointCallback(logdir=logdir)],
        logdir=logdir,
        verbose=True,
    )

mean_loss = 0
mean_normalized_loss = 0
n = 0.0

with torch.no_grad():
    for data in infer_loaders['infer']:
        n+=1.0
        x, y = data

        y_hat = net(x)
        
        loss = F.mse_loss(y_hat, y)
        zeros = torch.zeros_like(y)
        normalized_loss = F.mse_loss(y_hat, y) / F.mse_loss(y, zeros) 
        
        mean_loss += loss
        mean_normalized_loss += normalized_loss

mean_loss = mean_loss / n
mean_normalized_loss = mean_normalized_loss / n
"""
net.eval()
summary(net, input_size = (1, 1, 256, 256, 256), device='cuda', verbose=1)

#print("Loss: {}".format(mean_loss))
#print("Normalized loss: {}".format(mean_normalized_loss))

print("Done")