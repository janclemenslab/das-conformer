import numpy as np
import librosa
import soundfile

from typing import Optional, Dict, Sequence, Any
from torch.utils.data import Dataset
import torchaudio
import torch
from . import npy_dir
from torch.utils.data import DataLoader, random_split
import lightning as L
from pathlib import Path
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
