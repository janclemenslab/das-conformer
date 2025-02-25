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


class RawAudioDirSyllableGenerator(Dataset):
    def __init__(
        self,
        time_steps: int,
        audio_files: Sequence,
        annotation_files: Sequence,
        class_names: Sequence,
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
        self.audio_files = audio_files
        self.annotation_files = annotation_files
        self.class_names = class_names
        self.hop_s = hop_s

        self.audio = [soundfile.SoundFile(audio_file) for audio_file in self.audio_files]
        self.annotations = [pd.read_csv(annot_file) for annot_file in self.annotation_files]

        self.nb_classes = len(self.class_names)
        self.nb_samples_in_file = np.array([len(a) for a in self.audio])
        self.samplerate_per_file = np.array([a.samplerate for a in self.audio])

        self.nb_chunks_in_file = []

        for nb_samples, samplerate in zip(self.nb_samples_in_file, self.samplerate_per_file):
            chunk_duration = self.time_steps
            self.nb_chunks_in_file.append(int((nb_samples - chunk_duration) // chunk_duration))
        self.nb_chunks_in_file = np.array(self.nb_chunks_in_file)

        self.chunk_borders = np.concatenate(([0], np.cumsum(self.nb_chunks_in_file)))
        self.file_prob = self.nb_chunks_in_file / np.sum(self.nb_chunks_in_file)
        self.nb_total = int(np.sum(self.nb_chunks_in_file))

        # SoundFile not pickleable so delete here and recreate when calling __getitem__ the first time
        self.audio = None

    def __len__(self):
        return self.nb_total

    def __getitem__(self, idx):
        if self.audio is None:
            self.audio = [soundfile.SoundFile(audio_file) for audio_file in self.audio_files]

        id = max(0, np.argmax(self.chunk_borders >= idx) - 1)
        start = (idx - self.chunk_borders[id]) * self.time_steps

        stop = start + self.time_steps
        # a = soundfile.SoundFile(self.audio_files[id])
        a = self.audio[id]

        hop_samples = int(a.samplerate * self.hop_s)
        a.seek(start)
        chunk = a.read(frames=self.time_steps, dtype=np.float32)

        start_sec = start / a.samplerate
        stop_sec = stop / a.samplerate

        annot = self.annotations[id]
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


class RawAudioDirDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
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

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.nb_freq = nb_freq
        self.hop_s = hop_s
        # scan directory for files with annotations
        all_files = list(self.data_dir.glob("*"))
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.audio_files = []
        self.annotation_files = []
        for audio_file in all_files:
            try:
                soundfile.info(audio_file)
                annot_file = audio_file.parent / Path(str(audio_file.stem) + "_annotations.csv")
                if annot_file.exists():
                    self.audio_files.append(audio_file)
                    self.annotation_files.append(annot_file)
                else:
                    print(audio_file)
            except:
                pass

        # open all files
        # self.audios = [soundfile.SoundFile(wav_file) for wav_file in self.audio_files]
        self.annotations = [pd.read_csv(annot_file) for annot_file in self.annotation_files]

        # discover classes in annotations
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = []
            for annot in self.annotations:
                self.class_names.extend(list(set(annot["name"].to_list())))
            self.class_names = list(set(self.class_names))
        self.class_names.insert(0, "noise")
        self.nb_classes = len(self.class_names)

        # train/test/val split
        subsets = random_split(self.audio_files, [0.6, 0.2, 0.2])
        self.subsets = {}
        for subset, name in zip(subsets, ["train", "val", "test"]):
            self.subsets[name] = {
                "audio": [self.audio_files[i] for i in subset.indices],
                "annotations": [self.annotation_files[i] for i in subset.indices],
            }

    def setup(self, stage: str):
        pass

    def _dataloader(self, stage: str):
        if stage == "predict":  # do not use splits when predicting
            self.gen = RawAudioDirSyllableGenerator(
                time_steps=self.time_steps,
                audio_files=self.audio_files,
                annotation_files=self.annotation_files,
                class_names=self.class_names,
                nb_freq=self.nb_freq,
                hop_s=self.hop_s,
            )
        else: # use splits during train/val/test stage
            self.gen = RawAudioDirSyllableGenerator(
                time_steps=self.time_steps,
                audio_files=self.subsets[stage]["audio"],
                annotation_files=self.subsets[stage]["annotations"],
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
