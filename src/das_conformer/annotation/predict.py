"""Code for training and evaluating networks."""

import logging
import os
import shutil
import flammkuchen
import numpy as np

from .segment_utils import fill_gaps, remove_short, label_syllables_by_majority
from typing import List, Optional, Dict, Any, Sequence, Iterable, Union
import glob

# import tensorflow
import librosa

# import zarr
from tqdm.autonotebook import tqdm
# import dask.config
# import dask.array as da
# from dask.diagnostics import ProgressBar


def labels_from_probabilities(
    probabilities, threshold: Optional[float] = None, indices: Optional[Union[Sequence[int], slice]] = None
) -> np.ndarray:
    """Convert class-wise probabilities into labels.

    Args:
        probabilities ([type]): [samples, classes] or [samples, ]
        threshold (float, Optional): Argmax over all classes (Default, 2D - corresponds to 1/nb_classes or 0.5 if 1D).
                                     If float, each class probability is compared to the threshold.
                                     First class to cross threshold wins.
                                     If no class crosses threshold label will default to the first class.
        indices: (List[int], Optional): List of indices into axis 1 for which to compute the labels.
                                     Defaults to None (use all indices).
    Returns:
        labels [samples,] - index of "winning" dimension for each sample
    """
    if indices is None:
        indices = slice(None)  # equivalent to ":"

    if probabilities.ndim == 1:
        if threshold is None:
            threshold = 0.5
        labels = (probabilities[:, indices] > threshold).astype(np.intp)
    elif probabilities.ndim == 2:
        if threshold is None:
            labels = np.argmax(probabilities[:, indices], axis=1)
        else:
            thresholded_probabilities = probabilities[:, indices].copy()
            thresholded_probabilities[thresholded_probabilities < threshold] = 0
            labels = np.argmax(thresholded_probabilities > threshold, axis=1)
    else:
        raise ValueError(f"Probabilities have to many dimensions ({probabilities.ndim}). Can only be 1D or 2D.")

    return labels


def predict_segments(
    class_probabilities: np.ndarray,
    samplerate: float = 1.0,
    segment_dims: Optional[Sequence[int]] = None,
    segment_names: Optional[Sequence[str]] = None,
    segment_ref_onsets: Optional[List[float]] = None,
    segment_ref_offsets: Optional[List[float]] = None,
    segment_thres: float = 0.5,
    segment_minlen: Optional[float] = None,
    segment_fillgap: Optional[float] = None,
    segment_labels_by_majority: bool = True,
) -> Dict:
    """[summary]

    TODO: document different approaches for single-type vs. multi-type segment detection

    Args:
        class_probabilities ([type]): [T, nb_classes] with probabilities for each class and sample
                                      or [T,] with integer entries as class labels
        samplerate (float, optional): Hz. Defaults to 1.0.
        segment_dims (Optional[List[int]], optional): set of indices into class_probabilities corresponding
                                                      to segment-like song types.
                                                      Needs to include the noise dim.
                                                      Required to ignore event-like song types.
                                                      Defaults to None (all classes are considered segment-like).
        segment_names (Optional[List[str]], optional): Names for segment-like classes.
                                                       Defaults to None (use indices of segment-like classes).
        segment_ref_onsets (Optional[List[float]], optional):
                            Syllable onsets (in seconds) to use for estimating labels.
                            Defaults to None (will use onsets est from class_probabilitieslabels as ref).
        segment_ref_offsets (Optional[List[float]], optional): [description].
                            Syllable offsets (in seconds) to use for estimating labels.
                            Defaults to None (will use offsets est from class_probabilitieslabels as ref).
        segment_thres (float, optional): [description]. Defaults to 0.5.
        segment_minlen (Optional[float], optional): seconds. Defaults to None.
        segment_fillgap (Optional[float], optional): seconds. Defaults to None.
        segment_labels_by_majority (bool, optional): Segment labels given by majority of label values within on- and offsets.
                                                     Defaults to True.

    Returns:
        dict['segmentnames']['denselabels-samples'/'onsets'/'offsets'/'probabilities']
    """

    segments: Dict[str, Any] = dict()
    segments["samplerate_Hz"] = samplerate
    if segment_dims is None:
        segment_dims = np.arange(class_probabilities.shape[-1])
    segments["index"] = segment_dims
    segments["names"] = segment_names
    # if not probs_are_labels:
    segments["probabilities"] = class_probabilities[:, segment_dims]
    labels = labels_from_probabilities(class_probabilities, segment_thres, segment_dims)
    segments["labels"] = labels
    # turn into song (0), no song (1) sequence to detect onsets (0->1) and offsets (1->0)
    song_binary = (labels > 0).astype(np.int8)
    if segment_fillgap is not None:
        song_binary = fill_gaps(
            song_binary,
            gap_dur=int(segment_fillgap * samplerate),
        )
    if segment_minlen is not None:
        song_binary = remove_short(
            song_binary,
            min_len=int(segment_minlen * samplerate),
        )

    # detect syllable on- and offsets
    # pre- and post-pend 0 so we detect on and offsets at boundaries
    logging.info("   Detecting syllable on and offsets:")
    onsets = np.where(np.diff(song_binary, prepend=0) == 1)[0]
    offsets = np.where(np.diff(song_binary, append=0) == -1)[0]
    segments["onsets_seconds"] = onsets.astype(float) / samplerate
    segments["offsets_seconds"] = offsets.astype(float) / samplerate

    # there is just a single segment type plus noise - in that case we use the gap-filled, short-deleted pred
    sequence: List[int] = []  # default to empty list
    # if len(segment_dims) == 2:
    #     labels = song_binary
    #     # syllable-type for each syllable as int
    #     sequence = [str(segment_names[1])] * len(segments["offsets_seconds"])
    # # if >1 segment type (plus noise) label sylls by majority vote on un-smoothed labels
    # elif len(segment_dims) > 2 and
    if segment_labels_by_majority:
        # if no refs provided, use use on/offsets from smoothed labels
        if segment_ref_onsets is None:
            segment_ref_onsets = segments["onsets_seconds"]
        if segment_ref_offsets is None:
            segment_ref_offsets = segments["offsets_seconds"]

        logging.info("   Labeling by majority:")
        # if len(segment_dims) < np.iinfo("uint8").max:
        #     cast_to = np.uint8
        # elif len(segment_dims) < np.iinfo("uint16").max:
        #     cast_to = np.uint16
        # elif len(segment_dims) < np.iinfo("uint32").max:
        #     cast_to = np.uint32
        # else:
        #     cast_to = None

        # if cast_to is not None:
        #     logging.info(f"   Casting labels to {cast_to}:")
        #     labels = labels.astype(cast_to)

        # syllable-type for each syllable as int
        sequence, labels = label_syllables_by_majority(labels, segment_ref_onsets, segment_ref_offsets, samplerate)
    segments["samples"] = labels
    segments["sequence"] = sequence
    return segments
