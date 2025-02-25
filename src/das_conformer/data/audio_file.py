import numpy as np
import librosa
import soundfile

from typing import Optional, Dict, Sequence, Any
from torch.utils.data import Dataset
import torchaudio
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from pathlib import Path
import pandas as pd


class RawAudioSyllableGenerator(Dataset):
    def __init__(
        self,
        time_steps: int,
        audio_file: str,
        class_names: Sequence,
        annotation_file: Optional[str] = None,
        nb_freq: int = 128,
        hop_s: float = 0.002,
    ):
        """_summary_

        Args:
            time_steps (int): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            training (bool, optional): _description_. Defaults to True.
        """
        self.time_steps = time_steps
        self.audio_file = audio_file
        self.annotation_file = annotation_file
        self.annotations = None
        self.class_names = class_names
        self.hop_s = hop_s

        self.audio = soundfile.SoundFile(self.audio_file)
        if self.annotation_file is not None:
            self.annotations = pd.read_csv(self.annotation_file)

        self.nb_classes = len(self.class_names)
        self.nb_samples_in_file = len(self.audio)
        self.samplerate_per_file = self.audio.samplerate

        self.nb_chunks_in_file = []

        self.nb_chunks_in_file = int((self.nb_samples_in_file - self.time_steps) // self.time_steps)
        self.nb_total = int(self.nb_chunks_in_file)

        # SoundFile not pickleable so delete here and recreate when calling __getitem__ the first time
        self.audio = None

    def __len__(self):
        return self.nb_total

    def __getitem__(self, idx):
        if self.audio is None:
            self.audio = soundfile.SoundFile(self.audio_file)

        start = idx * self.time_steps
        stop = start + self.time_steps
        a = self.audio

        hop_samples = int(a.samplerate * self.hop_s)
        a.seek(start)
        chunk = a.read(frames=self.time_steps, dtype=np.float32)

        start_sec = start / a.samplerate
        stop_sec = stop / a.samplerate

        if self.annotations is None:
            return (
                chunk,
                self.time_steps,
            )
        else:
            annot = self.annotations
            annot = annot[np.logical_and(annot["start_seconds"] > start_sec, annot["start_seconds"] < stop_sec)]
            annot = annot[np.logical_and(annot["stop_seconds"] > start_sec, annot["stop_seconds"] < stop_sec)]

            stops = annot["stop_seconds"].values - start_sec
            starts = annot["start_seconds"].values - stop_sec
            names = annot["name"].values
            # print(self.time_steps, hop_samples, self.time_steps // hop_samples, a.samplerate)
            labels = np.zeros((self.time_steps // hop_samples + 1, self.nb_classes), dtype=np.float32)
            if len(stops) and len(starts):
                if stops[0] < starts[0]:
                    print(stops[0], starts[0])
                    starts = np.concatenate(([0], starts))

                if stops[-1] < starts[-1]:
                    print(stops[-1], starts[-1])
                    starts = np.append(starts, self.time_steps)

                starts = ((starts * a.samplerate) // hop_samples).astype(np.intp)
                stops = ((stops * a.samplerate) // hop_samples).astype(np.intp)

                for start, stop, name in zip(starts, stops, names):
                    if name in self.class_names:
                        labels[start:stop, self.class_names.index(name)] = 1.0

            labels[:, 0] = 1.0 - np.sum(labels[:, 1:], axis=1)

            return (
                chunk,
                self.time_steps,
                labels,
                self.time_steps,
            )


class RawAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        audio_file: str,
        batch_size: int = 32,
        time_steps: int = 512,
        nb_freq: int = 128,
        hop_s: float = 0.001,
        class_names: Optional[Sequence[str]] = None,
        num_workers: Optional[int] = 2,
        persistent_workers: bool = True,
    ):
        # TODO: Also acccept folder with files - make npy_dir or run inference directly

        super().__init__()
        self.save_hyperparameters()

        # self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.nb_freq = nb_freq
        self.hop_s = hop_s
        # scan directory for files with annotations
        # all_files = list(self.data_dir.glob("*"))
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.audio_file = audio_file
        self.annotation_file = None
        self.annotations = None
        soundfile.info(audio_file)
        annot_file = audio_file.parent / Path(str(audio_file.stem) + "_annotations.csv")
        if annot_file.exists():
            self.annotation_file = annot_file
            self.annotations = pd.read_csv(self.annotation_file)

        # discover classes in annotations
        self.class_names = class_names
        if self.class_names is None and self.annotation_file is not None:
            self.class_names = list(set(self.annotations["name"].to_list()))
        self.class_names.insert(0, "noise")
        self.nb_classes = len(self.class_names)

        # # train/test/val split
        # subsets = random_split(self.audio_files, [0.6, 0.2, 0.2])
        # self.subsets = {}
        # for subset, name in zip(subsets, ["train", "val", "test"]):
        #     self.subsets[name] = {
        #         "audio": [self.audio_files[i] for i in subset.indices],
        #         "annotations": [self.annotation_files[i] for i in subset.indices],
        #     }

    def setup(self, stage: str):
        pass

    def _dataloader(self, stage: str):
        self.gen = RawAudioSyllableGenerator(
            time_steps=self.time_steps,
            audio_file=self.audio_file,
            annotation_file=self.annotation_file,
            class_names=self.class_names,
            nb_freq=self.nb_freq,
            hop_s=self.hop_s,
        )

        return DataLoader(
            self.gen,
            batch_size=self.batch_size,
            shuffle=stage == "train",
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")  # DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
