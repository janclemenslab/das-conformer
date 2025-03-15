import numpy as np
from typing import Optional, Dict, Sequence, Any
from torch.utils.data import Dataset
import torchaudio
import torch
from .. import npy_dir
from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
import soundfile
import pandas as pd


class SyllableGenerator(Dataset):
    def __init__(
        self,
        time_steps: int,
        hop_length: int,
        specs: Sequence,
        labels: Sequence,
        binary: bool = False,
        augs=None,
    ):
        """
        A dataset generator for syllable-level audio data.

        Args:
            time_steps (int): The number of time steps in each audio chunk.
            hop_length (int): The number of frames to skip between consecutive chunks.
            specs (Sequence): A sequence of spectrograms or audio features.
            labels (Sequence): A sequence of corresponding labels.
            binary (bool, optional): If True, convert labels to binary format. Defaults to False.

        """
        self.time_steps = time_steps
        self.hop_length = hop_length
        self.specs = specs
        self.labels = labels
        self.binary = binary
        self.augs = augs
        self.nb_total = (len(self.labels) - self.time_steps - 1) // self.time_steps

    def __len__(self):
        return self.nb_total

    def __getitem__(self, idx):
        start = int(idx * self.time_steps)
        stop = start + self.time_steps + 1
        audio_chunk = self.specs[start:stop].T.astype(np.float32)
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk[np.newaxis, :]

        if not self.binary:
            y = self.labels[start : stop : self.hop_length, :].astype(np.float32)
        else:
            y0 = self.labels[start : stop : self.hop_length, :].astype(np.float32)
            y = np.zeros((y0.shape[0], 2), dtype=np.float32)
            y[:, 0] = y0[:, 0]
            y[:, 1] = 1.0 - y0[:, 0]

        if self.augs is not None:
            audio_chunk = self.augs(
                torch.tensor(audio_chunk[:, None, :]), sample_rate=self.augs.sample_rate
            )[:, 0, :]

        return (
            audio_chunk,
            self.time_steps,
            y,
            self.time_steps,
        )


class RawNPYDirDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        time_steps: int = 512,
        nb_freq: int = 128,
        hop_length: int = 512,
        train_repeats: int = 1,
        nb_workers: int = 2,
        binary: bool = False,
        augs=None,
        training: bool = False,
    ):
        """
        Initialize the RawNPYDirDataModule.

        Args:
            data_dir (str): Directory containing the NPY data files.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            time_steps (int, optional): Number of time steps in each audio chunk. Defaults to 512.
            nb_freq (int, optional): Number of frequency bins. Defaults to 128.
            hop_length (int, optional): Number of frames to skip between consecutive chunks. Defaults to 512.
            train_repeats (int, optional): Number of times to repeat the training data. Defaults to 1.
            nb_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
            binary (bool, optional): If True, convert labels to binary format [0, 1], e.g. noise vs. syllable, ignoring syllable types. Defaults to False.
        """
        # TODO: Also acccept folder with files - make npy_dir or run inference directly

        super().__init__()
        self.save_hyperparameters()
        print("test")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.hop_length = hop_length
        self.train_repeats = train_repeats
        self.nb_workers = nb_workers
        self.binary = binary
        self.augs = augs
        self.training = training
        self.data = npy_dir.load(self.data_dir, memmap_dirs="all")

    def setup(self, stage: str):
        pass

    def _dataloader(self, stage):
        gen = SyllableGenerator(
            time_steps=self.time_steps,
            hop_length=self.hop_length,
            specs=self.data[stage]["x"],
            labels=self.data[stage]["y"],
            binary=self.binary,
            augs=self.augs if stage == "train" else None,
        )

        return DataLoader(
            gen,
            batch_size=self.batch_size,
            shuffle=(stage == "train") and self.training,
            num_workers=self.nb_workers,
            persistent_workers=self.nb_workers > 0,
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
