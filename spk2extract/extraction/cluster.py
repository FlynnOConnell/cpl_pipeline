"""
Cluster and extract function utilities.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from numba import jit, njit
from numba.typed import List
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.signal import sosfilt


def group_consecutives(arr):
    """
    Group consecutive numbers in an array into a list of arrays.

    Parameters
    ----------
    arr : array-like
        Input array.

    Returns
    -------
    out : list of array-like
        List of arrays of consecutive numbers.

    """

    diff_arr = np.diff(arr)
    change = np.where(diff_arr > 1)[0] + 1
    out = []
    prev = 0
    for i in change:
        out.append(arr[prev:i])
        prev = i

    out.append(arr[prev:])
    return out


def filter_signal(
    signal,
    sampling_rate: int | float,
    freq,
):
    """
    Apply a bandpass filter to the input electrode signal using a Butterworth digital and analog filter design.

    Parameters
    ----------
    signal : array-like
        The input electrode signal as a 1-D array.
    sampling_rate : float
        The sampling rate of the signal in Hz. Default is 20kHz.
    freq : tuple
        The frequency range for the bandpass filter as (low_frequency, high_frequency).

    Returns
    -------
    filt_el : array-like
        The filtered electrode signal as a 1-D array.
    """

    # noinspection
    m, n = butter(
        2,
        [2.0 * freq[0] / sampling_rate, 2.0 * freq[1] / sampling_rate],
        btype="bandpass",
    )
    filt_el = filtfilt(m, n, signal)
    return filt_el


def get_detection_threshold(filt_el):
    """
    Calculates the spike detection threshold as 5x the median absolute deviation

    Parameters
    ----------
    filt_el : np.array, 1-D
        filtered electrode trace

    Returns
    -------
    threshold : float
        spike detection threshold

    """
    m = np.mean(filt_el)
    th = 5.0 * np.median(np.abs(filt_el) / 0.6745)
    return m - th


def detect_spikes(filt_el, spike_snapshot, fs, thresh=None):
    """
    Detects spikes in the filtered electrode trace and return the waveforms
    and spike_times

    Parameters
    ----------
    filt_el : np.array, 1-D
        filtered electrode trace
    spike_snapshot : tuple,
        2-elements, [ms before spike minimum, ms after spike minimum]
        time around spike to snap as waveform
    fs : float,
        sampling rate in Hz
    thresh : float, optional
        spike detection threshold, if None, calculated as 5x median absolute deviation

    Returns
    -------
    waves : np.array
        matrix of de-jittered, spike waveforms, upsampled by 10x, row for each spike
    times : np.array
        array of spike times in samples
    threshold: float
        spike detection threshold
    """
    # get indices of spike snapshot, expand by .1 ms in each direction
    snapshot = np.arange(
        -(spike_snapshot[0] + 0.1) * fs / 1000,
        1 + (spike_snapshot[1] + 0.1) * fs / 1000,
    ).astype("int64")
    if thresh is None:
        thresh = get_detection_threshold(filt_el)

    pos = np.where(filt_el <= thresh)[0]
    consecutive = group_consecutives(pos)

    waves = []
    times = []
    for idx in consecutive:
        minimum = idx[np.argmin(filt_el[idx])]
        spike_idx = minimum + snapshot
        if spike_idx[0] >= 0 and spike_idx[-1] < len(filt_el):
            waves.append(filt_el[spike_idx])
            times.append(minimum)

    if len(waves) == 0:
        return None, None

    waves_dj, times_dj = dejitter(np.array(waves), np.array(times), spike_snapshot, fs)
    return waves_dj, times_dj, thresh


def dejitter(
    slices: list[np.ndarray] | np.ndarray, spike_times, spike_snapshot, sampling_rate
):
    """
    Adjust the alignment of extracted spike waveforms to minimize jitter.

    Parameters
    ----------
    slices : np.ndarray or list of np.ndarray
        Extracted spike waveforms, each as a 1-D array. Can be a list of arrays or a 2-D array.
    spike_times : list of int
        List of indices indicating the positions of the extracted spikes in the input array.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    spike_snapshot : tuple of float, optiona
        The time range (in milliseconds) around each spike to extract, given as (pre_spike_time, post_spike_time).

    Returns
    -------
    slices_dejittered : array-like
        The dejittered spike waveforms as a 2-D array.
    spike_times_dejittered : array-like
        The updated spike times as a 1-D array.
    """
    x = np.arange(0, len(slices[0]), 1)
    xnew = np.arange(0, len(slices[0]) - 1, 0.1)

    # Calculate the number of samples to be sliced out around each spike's minimum
    before = int((sampling_rate / 1000.0) * (spike_snapshot[0]))
    after = int((sampling_rate / 1000.0) * (spike_snapshot[1]))

    slices_dejittered = []
    spike_times_dejittered = []
    for i in range(len(slices)):
        f = interp1d(x, slices[i])
        # 10-fold interpolated spike
        ynew = f(xnew)
        orig_min = np.where(slices[i] == np.min(slices[i]))[0][0]
        orig_min_time = x[orig_min] / (sampling_rate / 1000)
        minimum = np.where(ynew == np.min(ynew))[0][0]
        min_time = xnew[minimum] / (sampling_rate / 1000)
        # Only accept spikes if the interpolated minimum has shifted by
        # less than 1/10th of a ms (3 samples for a 30kHz recording, 30
        # samples after interpolation)
        if np.abs(min_time - orig_min_time) <= 0.1:
            # If minimum is too close to the end for a full snapshot then toss out spike
            if minimum + after * 10 < len(ynew) and minimum - before * 10 >= 0:
                slices_dejittered.append(
                    ynew[minimum - before * 10 : minimum + after * 10]
                )
                spike_times_dejittered.append(spike_times[i])
    return np.array(slices_dejittered), np.array(spike_times_dejittered)
