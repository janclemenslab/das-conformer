import numpy as np
from typing import Optional, Dict, Sequence, Any
from torch.utils.data import Dataset
import torchaudio
import torch
from . import npy_dir
from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
import soundfile
import pandas as pd


class SyllableGenerator(Dataset):
    def __init__(
        self,
        time_steps: int,
        specs: Sequence,
        labels: Sequence,
        batch_size: int = 32,
        training: bool = True,
        repeats: int = 1,
        augment_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """_summary_

        Args:
            time_steps (int): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            training (bool, optional): _description_. Defaults to True.
            augment_kwargs (Optional[Dict[str, Any]], optional): See args to `augment`. Defaults to None.
        """
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.training = training
        self.labels = labels
        self.specs = specs
        self.repeats = repeats
        self.nb_total = (self.repeats * len(self.labels)) // self.time_steps

    def __len__(self):
        return self.nb_total

    def __getitem__(self, idx):
        # if idx is None:
        if self.training:
            start = np.random.choice(self.labels.shape[0] - self.time_steps - 1)
        else:
            start = int(idx * self.time_steps)
        stop = start + self.time_steps
        return (
            self.specs[start:stop, :],
            self.time_steps,
            self.labels[start:stop, :],
            self.time_steps,
        )


class NPYDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, time_steps: int = 512, nb_freq: int = 128):
        # TODO: Also acccept folder with files - make npy_dir or run inference directly

        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.data = npy_dir.load(self.data_dir)


    def _dataloader(self, stage: str):
        gen = SyllableGenerator(
            time_steps=self.time_steps,
            specs=self.data[stage]["x_spec"].T,
            labels=self.data[stage]["y_spec"],
            training=False,
        )

        return DataLoader(
            gen,
            batch_size=self.batch_size,
            shuffle=stage == "train",
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("test")

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
