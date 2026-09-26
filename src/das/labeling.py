from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config
from .data.audio_dir import _ANNOTATION_BOUNDARY_NAMES
from .prediction_results import write_annotation_file


@dataclass
class LabelAudioEntry:
    audio: np.ndarray
    samplerate: int
    filepath: Path | None = None
    annotations: pd.DataFrame | None = None


@dataclass
class LabelingResult:
    annotations: pd.DataFrame | list[pd.DataFrame]
    items: pd.DataFrame
    features: np.ndarray
    embedding: np.ndarray
    cluster_labels: np.ndarray
    audio_paths: list[Path | None]


def run_labeling(entries: Sequence[LabelAudioEntry], config: Config) -> LabelingResult:
    entries = list(entries)
    if not entries:
        raise ValueError("No audio inputs were provided for labeling.")

    items = collect_label_items(entries, config)
    if items.empty:
        raise ValueError("No labeling items were created.")

    features = compute_features(entries, items, config)
    embedding = compute_embedding(features, config)
    cluster_labels = cluster_embedding(embedding, config)
    annotations = annotations_from_clusters(items, cluster_labels, audio_count=len(entries))
    return LabelingResult(
        annotations=annotations[0] if len(annotations) == 1 else annotations,
        items=items.assign(cluster_label=cluster_labels),
        features=features,
        embedding=embedding,
        cluster_labels=cluster_labels,
        audio_paths=[entry.filepath for entry in entries],
    )


def recluster_result(result: LabelingResult, config: Config) -> LabelingResult:
    cluster_labels = cluster_embedding(result.embedding, config)
    annotations = annotations_from_clusters(result.items, cluster_labels, audio_count=len(result.audio_paths))
    return LabelingResult(
        annotations=annotations[0] if len(annotations) == 1 else annotations,
        items=result.items.drop(columns=["cluster_label"], errors="ignore").assign(cluster_label=cluster_labels),
        features=result.features,
        embedding=result.embedding,
        cluster_labels=cluster_labels,
        audio_paths=list(result.audio_paths),
    )


def write_label_outputs(result: LabelingResult, *, output_dir: Path, output_suffix: str, merge: bool = False) -> list[str]:
    annotations = result.annotations if isinstance(result.annotations, list) else [result.annotations]
    output_paths = []
    for audio_path, frame in zip(result.audio_paths, annotations, strict=True):
        if audio_path is None:
            raise ValueError("Cannot write raw-audio labeling outputs without an audio filepath.")
        output_path = Path(output_dir) / f"{Path(audio_path).stem}{output_suffix}"
        write_annotation_file(output_path, frame, merge=merge)
        output_paths.append(str(output_path))
    return output_paths


def collect_label_items(entries: Sequence[LabelAudioEntry], config: Config) -> pd.DataFrame:
    if config.label_items == "windows":
        return _collect_window_items(entries, config)
    return _collect_segment_items(entries, config)


def compute_features(entries: Sequence[LabelAudioEntry], items: pd.DataFrame, config: Config) -> np.ndarray:
    rows = []
    for row in items.itertuples(index=False):
        entry = entries[int(row.audio_index)]
        audio = _mono_audio(entry.audio)
        samplerate = int(entry.samplerate)
        start_sample = max(0, int(round(float(row.start_seconds) * samplerate)))
        stop_sample = min(len(audio), int(round(float(row.stop_seconds) * samplerate)))
        if stop_sample <= start_sample:
            stop_sample = min(len(audio), start_sample + 1)
        rows.append(_spectrogram_features(audio[start_sample:stop_sample], samplerate, config).reshape(-1))
    return np.vstack(rows).astype(np.float32)


def compute_embedding(features: np.ndarray, config: Config) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array.")
    if features.shape[0] < 3:
        return _svd_embedding(features)
    if config.label_embedding == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(float(config.label_tsne_perplexity), max(1.0, float(features.shape[0] - 1)))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=config.label_random_state,
            init="random",
            learning_rate="auto",
        ).fit_transform(features).astype(np.float32)

    try:
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError("UMAP labeling requires the optional dependency `umap-learn`. Install `das[labeling]`.") from exc

    n_neighbors = min(int(config.label_umap_n_neighbors), max(2, features.shape[0] - 1))
    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(config.label_umap_min_dist),
        random_state=config.label_random_state,
    ).fit_transform(features).astype(np.float32)


def cluster_embedding(embedding: np.ndarray, config: Config) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        raise ValueError("embedding must be a non-empty 2D array.")
    min_cluster_size = int(config.label_hdbscan_min_cluster_size)
    if embedding.shape[0] < min_cluster_size:
        return np.full(embedding.shape[0], -1, dtype=np.int64)
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError("Labeling requires sklearn.cluster.HDBSCAN from scikit-learn.") from exc

    kwargs = {
        "min_cluster_size": min_cluster_size,
        "cluster_selection_epsilon": float(config.label_hdbscan_cluster_selection_epsilon),
    }
    if config.label_hdbscan_min_samples is not None:
        kwargs["min_samples"] = int(config.label_hdbscan_min_samples)
    return np.asarray(HDBSCAN(**kwargs).fit_predict(embedding), dtype=np.int64)


def annotations_from_clusters(items: pd.DataFrame, cluster_labels: np.ndarray, *, audio_count: int) -> list[pd.DataFrame]:
    rows_by_audio = [[] for _ in range(audio_count)]
    for item, cluster_label in zip(items.itertuples(index=False), cluster_labels, strict=True):
        cluster_label = int(cluster_label)
        if cluster_label < 0:
            continue
        rows_by_audio[int(item.audio_index)].append(
            {
                "name": f"cluster_{cluster_label}",
                "start_seconds": float(item.start_seconds),
                "stop_seconds": float(item.stop_seconds),
            }
        )
    columns = ["name", "start_seconds", "stop_seconds"]
    return [pd.DataFrame(rows, columns=columns) for rows in rows_by_audio]


def _collect_segment_items(entries: Sequence[LabelAudioEntry], config: Config) -> pd.DataFrame:
    rows = []
    include_labels = set(config.include_labels)
    fallback_window = float(config.label_window_seconds)
    for audio_index, entry in enumerate(entries):
        if entry.annotations is None:
            raise ValueError("label_items='segments' requires annotations.")
        annotations = entry.annotations
        if annotations.empty:
            continue
        for row in annotations.itertuples(index=False):
            name = str(row.name)
            if name in _ANNOTATION_BOUNDARY_NAMES:
                continue
            if include_labels and name not in include_labels:
                continue
            start = float(row.start_seconds)
            stop = float(row.stop_seconds)
            if stop <= start:
                half = fallback_window / 2.0
                start = max(0.0, start - half)
                stop = start + fallback_window
            rows.append(
                {
                    "audio_index": audio_index,
                    "filepath": "" if entry.filepath is None else str(entry.filepath),
                    "start_seconds": start,
                    "stop_seconds": stop,
                    "true_label": name,
                }
            )
    return pd.DataFrame(rows, columns=["audio_index", "filepath", "start_seconds", "stop_seconds", "true_label"])


def _collect_window_items(entries: Sequence[LabelAudioEntry], config: Config) -> pd.DataFrame:
    rows = []
    window = float(config.label_window_seconds)
    stride = float(config.label_window_stride_seconds)
    for audio_index, entry in enumerate(entries):
        duration = len(_mono_audio(entry.audio)) / float(entry.samplerate)
        if duration <= 0:
            continue
        starts = [0.0] if duration <= window else list(np.arange(0.0, duration - window + 1e-12, stride))
        for start in starts:
            stop = min(float(start) + window, duration)
            rows.append(
                {
                    "audio_index": audio_index,
                    "filepath": "" if entry.filepath is None else str(entry.filepath),
                    "start_seconds": float(start),
                    "stop_seconds": float(stop),
                    "true_label": "",
                }
            )
    return pd.DataFrame(rows, columns=["audio_index", "filepath", "start_seconds", "stop_seconds", "true_label"])


def _spectrogram_features(audio: np.ndarray, samplerate: int, config: Config) -> np.ndarray:
    from scipy.signal import stft

    audio = _mono_audio(audio)
    n_fft = int(config.frontend_kernel_size or 1024)
    hop = max(1, int(round(float(config.frontend_hop_seconds or 0.004) * samplerate)))
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    freqs, _times, zxx = stft(
        audio,
        fs=int(samplerate),
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop),
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx).astype(np.float32) ** 2
    if str(config.frontend_type) == "stft":
        spec = _select_frequency_bounds(power, freqs, config)
    else:
        spec = _mel_spectrogram(power, samplerate, n_fft, config)
    spec = _resize_time(spec, int(config.label_time_bins))
    if bool(config.label_log_scale):
        spec = np.log1p(spec)
    if bool(config.label_amplitude_normalize):
        denom = np.percentile(np.abs(spec), 99)
        if denom > 0:
            spec = spec / denom
    return spec.astype(np.float32)


def _mel_spectrogram(power: np.ndarray, samplerate: int, n_fft: int, config: Config) -> np.ndarray:
    import librosa

    n_mels = int(config.frontend_num_channels or 128)
    fmax = config.frontend_fmax if config.frontend_fmax is not None else samplerate / 2.0
    mel = librosa.filters.mel(
        sr=int(samplerate),
        n_fft=int(n_fft),
        n_mels=n_mels,
        fmin=float(config.frontend_fmin),
        fmax=float(fmax),
    )
    return np.asarray(mel @ power, dtype=np.float32)


def _select_frequency_bounds(power: np.ndarray, freqs: np.ndarray, config: Config) -> np.ndarray:
    fmax = config.frontend_fmax if config.frontend_fmax is not None else np.inf
    keep = np.logical_and(freqs >= float(config.frontend_fmin), freqs <= float(fmax))
    if not np.any(keep):
        raise ValueError("No spectrogram frequency bins remain after applying frontend_fmin/frontend_fmax.")
    return np.asarray(power[keep], dtype=np.float32)


def _resize_time(spec: np.ndarray, time_bins: int) -> np.ndarray:
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape[1] == time_bins:
        return spec
    if spec.shape[1] == 0:
        return np.zeros((spec.shape[0], time_bins), dtype=np.float32)
    if spec.shape[1] == 1:
        return np.repeat(spec, time_bins, axis=1)
    source = np.linspace(0.0, 1.0, spec.shape[1])
    target = np.linspace(0.0, 1.0, time_bins)
    return np.vstack([np.interp(target, source, row) for row in spec]).astype(np.float32)


def _svd_embedding(features: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float32) - np.mean(features, axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    embedding = u[:, :2] * s[:2]
    if embedding.shape[1] == 1:
        embedding = np.column_stack([embedding[:, 0], np.zeros(embedding.shape[0], dtype=np.float32)])
    return embedding.astype(np.float32)


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 32 < audio.shape[1]:
            return np.mean(audio, axis=0).astype(np.float32)
        return np.mean(audio, axis=1).astype(np.float32)
    raise ValueError("Audio must be a 1D mono or 2D channel-by-time array.")
