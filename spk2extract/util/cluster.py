"""
Cluster and extract function utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numba import jit, njit
from numba.typed import List
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.signal import sosfilt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logpath = Path().home() / "autosort" / "cluster.log"


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


@njit
def levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between 2 sequences.

    Parameters
    ----------
    seq1 : array-like
        First sequence.
    seq2 : array-like
        Second sequence.

    Returns
    -------
    matrix[size_x - 1, size_y - 1] : float
        Levenshtein distance between the two sequences.

    """

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x

    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )

    return matrix[size_x - 1, size_y - 1]


@njit
def euclidean(a, b):
    """
    Computes Euclidean distance between 2 sequences.

    Parameters
    ----------
    a : array-like
        First sequence.
    b : array-like
        Second sequence.

    Returns
    -------
    c : float
        Euclidean distance between the two sequences.

    """
    c = np.power(a - b, 2)
    return np.sqrt(np.sum(c))


def filter_signal(
    sig,
    sampling_rate: int | float,
    freq=(300, 6000),
):
    """
    Apply a bandpass filter to the input electrode signal using a Butterworth digital and analog filter design.

    Parameters
    ----------
    sig : array-like
        The input electrode signal as a 1-D array.
    sampling_rate : float, optional
        The sampling rate of the signal in Hz. Default is 20kHz.
    freq : tuple, optional
        The frequency range for the bandpass filter as (low_frequency, high_frequency).
        Default is (300, 6000.0).

    Returns
    -------
    filt_el : array-like
        The filtered electrode signal as a 1-D array.
    """
    if freq is None:
        freq = (300, 6000)

    sos = butter(
        2,
        [2.0 * freq[0] / sampling_rate, 2.0 * freq[1] / sampling_rate],
        btype="bandpass",
        output="sos",
    )
    filt_el = sosfilt(sos, sig)
    return filt_el


@jit(nopython=True)
def extract_waveforms(
    signal: np.ndarray,
    sampling_rate: int | float,
    spike_snapshot: tuple,
    STD=2.0,
    cutoff_std=10.0,
):
    """
    Extract individual spike waveforms from the filtered electrode signal.

    Parameters
    ----------
    signal : array-like
        The (already bandpass-filtered) electrode signal as a 1-D array.
    sampling_rate : float
        The sampling rate of the signal in Hz. Default is 20000.0 Hz.
    spike_snapshot : tuple of float
        The time range (in milliseconds) around each spike to extract, given as (pre_spike_time, post_spike_time).
    STD : float, optional
        The number of standard deviations to use for the spike detection threshold. Default is 2.0.
    cutoff_std : float, optional
        The cutoff threshold, in units of the standard deviation, for rejecting spikes. Default is 10.0.

    Returns
    -------
    slices : list of array-like
        List of extracted spike waveforms, each as a 1-D array. 2D, with slices[0] being the first spike, etc.
    spike_times : list of int
        List of indices indicating the positions of the extracted spikes in the input array.

    .. note::

        To get the number of extracted waveforms, use slices.shape[0].
        To get the number of samples in each waveform, use slices.shape[1].

        For spike_times, use len(spike_times), which should be the same as slices.shape[0].
    """
    print(spike_snapshot)
    m = np.mean(signal)
    th = np.std(signal) * STD
    pos = np.where(signal <= m - th)[0]
    changes = List()
    for i in range(len(pos) - 1):
        if pos[i + 1] - pos[i] > 1:
            changes.append(i + 1)

    slices = List()
    spike_times = List()
    for i in range(len(changes) - 1):
        minimum = np.where(
            signal[pos[changes[i] : changes[i + 1]]]
            == np.min(signal[pos[changes[i] : changes[i + 1]]])
        )[0]
        if pos[minimum[0] + changes[i]] - int(
            (spike_snapshot[0] + 0.1) * (sampling_rate / 1000.0)
        ) > 0 and pos[minimum[0] + changes[i]] + int(
            (spike_snapshot[1] + 0.1) * (sampling_rate / 1000.0)
        ) < len(
            signal
        ):
            tempslice = signal[
                pos[minimum[0] + changes[i]]
                - int((spike_snapshot[0] + 0.1) * (sampling_rate / 1000.0)) : pos[
                    minimum[0] + changes[i]
                ]
                + int((spike_snapshot[1] + 0.1) * (sampling_rate / 1000.0))
            ]
            if ~np.any(np.absolute(tempslice) > (th * cutoff_std) / STD):
                slices.append(tempslice)
                spike_times.append(pos[minimum[0] + changes[i]])
    assert len(slices) == len(spike_times)
    return slices, spike_times


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


def detect_spikes(filt_el, spike_snapshot=(0.5, 1.0), fs=30000.0, thresh=None):
    """
    Detects spikes in the filtered electrode trace and return the waveforms
    and spike_times

    Parameters
    ----------
    filt_el : np.array, 1-D
        filtered electrode trace
    spike_snapshot : list
        2-elements, [ms before spike minimum, ms after spike minimum]
        time around spike to snap as waveform
    fs : float, sampling rate in Hz
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


def get_filtered_electrode(data, freq=(300.0, 3000.0), sampling_rate=30000.0):
    """
    Apply a bandpass filter to the input electrode signal using a Butterworth digital and analog filter design.

    Parameters
    ----------
    data : array-like
        The input electrode signal as a 1-D array.
    sampling_rate : float, optional
        The sampling rate of the signal in Hz. Default is 20kHz.
    freq : tuple, optional
        The frequency range for the bandpass filter as (low_frequency, high_frequency).
        Default is (300, 6000.0).

    Returns
    -------
    filt_el : array-like
        The filtered electrode signal as a 1-D array.

    """
    el = data

    m, n, _ = butter(
        2,
        [2.0 * freq[0] / sampling_rate, 2.0 * freq[1] / sampling_rate],
        btype="bandpass",
    )
    filt_el = filtfilt(m, n, el)
    return filt_el


def dejitter(slices, spike_times, spike_snapshot=(0.5, 1.0), sampling_rate=30000.0):
    """
    Adjust the alignment of extracted spike waveforms to minimize jitter.

    Parameters
    ----------
    slices : list of array-like
        List of extracted spike waveforms, each as a 1-D array.
    spike_times : list of int
        List of indices indicating the positions of the extracted spikes in the input array.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    spike_snapshot : tuple of float, optional
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


def get_waveforms(
    el_trace,
    spike_times,
    snapshot=[0.5, 1.0],
    sampling_rate=30000.0,
    bandpass=[300, 3000],
):
    filt_el = get_filtered_electrode(
        el_trace, freq=bandpass, sampling_rate=sampling_rate
    )
    del el_trace
    pre_pts = int((snapshot[0] + 0.1) * (sampling_rate / 1000))
    post_pts = int((snapshot[1] + 0.2) * (sampling_rate / 1000))
    slices = np.zeros((spike_times.shape[0], pre_pts + post_pts))
    for i, st in enumerate(spike_times):
        slices[i, :] = filt_el[st - pre_pts : st + post_pts]
    slices_dj, times_dj = dejitter(slices, spike_times, snapshot, sampling_rate)
    return slices_dj, sampling_rate * 10
