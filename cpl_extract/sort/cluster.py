from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import umap
from scipy import linalg
from scipy.signal import find_peaks
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from statsmodels.stats.diagnostic import lilliefors
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from cpl_extract.sort.plot_blechpy import (
    plot_waveforms,
    plot_ISIs,
    plot_cluster_features,
    plot_mahalanobis_to_cluster, plot_recording_cutoff, plot_explained_pca_variance,
)
from cpl_extract.sort.spk_config import SortConfig
from cpl_extract.spk_io import print_dict, h5io
from cpl_extract.spk_io.writer import (
    write_dict_to_json,
    write_pandas_to_table,
    read_dict_from_json,
    read_pandas_from_table,
)


def get_filtered_electrode(data, freq=[300.0, 3000.0], sampling_rate=30000.0):
    el = data
    m, n = butter(
        2,
        [2.0 * freq[0] / sampling_rate, 2.0 * freq[1] / sampling_rate],
        btype="bandpass",
    )
    filt_el = filtfilt(m, n, el)
    return filt_el


def get_waveforms(
    el_trace,
    spike_times,
    snapshot=[0.5, 1.0],
    sampling_rate=30000.0,
    bandpass=[300, 3000],
):
    # Filter and extract waveforms
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


def implement_wavelet_transform(waves, n_pc=10):
    coeffs = pywt.wavedec(waves, "haar", axis=1)
    all_coeffs = np.column_stack(coeffs)
    k_stats = np.zeros((all_coeffs.shape[1],))
    p_vals = np.ones((all_coeffs.shape[1],))
    for i, c in enumerate(all_coeffs.T):
        k_stats[i], p_vals[i] = lilliefors(c, dist="norm")

    idx = np.argsort(p_vals)
    return all_coeffs[:, idx[:n_pc]]


def implement_pca(scaled_slices):
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)
    return pca_slices, pca.explained_variance_ratio_


def implement_umap(waves, n_pc=3, n_neighbors=30, min_dist=0.0):
    reducer = umap.UMAP(n_components=n_pc, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(waves)


def get_waveform_energy(waves):
    """
    Returns array of waveform energies

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    """
    return np.sqrt(np.sum(waves**2, axis=1)) / waves.shape[1]


def get_spike_slopes(waves):
    """
    Returns array of spike slopes (initial downward slope of spike)

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    """
    slopes = np.zeros((waves.shape[0],))
    for i, wave in enumerate(waves):
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        slopes[i] = (wave[minima] - wave[maxima]) / (minima - maxima)

    return slopes


def get_ISI_and_violations(
    spike_times,
    fs,
):
    """
    Returns array of inter-spike-intervals (ms) and corresponding number of 1ms and 2ms violations.

    Parameters
    ----------
    spike_times : numpy.array
    fs : float, sampling rate in Hz

    Returns
    -------
    isi : numpy.array
        interspike-intervals for the given spike-times
    violations1 : int
        number of 1ms violations
    violations2 : int
        number of 2ms violations
    """
    fs = float(fs / 1000.0)
    isi = np.ediff1d(np.sort(spike_times)) / fs
    violations1 = np.sum(isi < 1.0)
    violations2 = np.sum(isi < 2.0)

    return isi, violations1, violations2


def scale_waveforms(waves, energy=None):
    """
    Scale the extracted spike waveforms by their energy.

    Parameters
    ----------
    waves : array-like
        The spike waveforms as a 2-D array.
    energy : array-like
        The energy of each spike waveform as a 1-D array.

    Returns
    -------
    scaled_slices : array-like
        The scaled spike waveforms as a 2-D array.
    """
    if energy is None:
        energy = get_waveform_energy(waves)
    elif len(energy) != waves.shape[0]:
        raise ValueError(
            (
                "Energies must correspond to each waveforms."
                "Different lengths are not allowed"
            )
        )

    scaled_slices = np.zeros(waves.shape)
    for i, w in enumerate(zip(waves, energy)):
        scaled_slices[i] = w[0] / w[1]

    return scaled_slices


def compute_waveform_metrics(waves, n_pc=3, use_umap=False):
    """

    Make clustering data array with columns:
         - amplitudes, energy, slope, pc1, pc2, pc3, etc.
    Parameters
    ----------
    waves : np.array
        waveforms with a row for each spike waveform
    n_pc : int (optional)
        number of principal components to include in data array
    use_umap : bool (optional)
        whether to use UMAP instead of PCA for dimensionality reduction

    Returns
    -------
    np.array
    """
    data = np.zeros((waves.shape[0], 3))
    for i, wave in enumerate(waves):
        data[i, 0] = np.min(wave)
        data[i, 1] = np.sqrt(np.sum(wave**2)) / len(wave)
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        data[i, 2] = (wave[minima] - wave[maxima]) / (minima - maxima)

    # Scale waveforms to energy before running PCA
    if use_umap:
        pc_waves = implement_umap(waves, n_pc=n_pc)
    else:
        scaled_waves = scale_waveforms(waves, energy=data[:, 1])
        pc_waves, _ = implement_pca(scaled_waves)

    data = np.hstack((data, pc_waves[:, :n_pc]))
    data_columns = ["amplitude", "energy", "spike_slope"]
    data_columns.extend(["PC%i" % i for i in range(n_pc)])
    return data, data_columns


def get_mahalanobis_distances_to_cluster(data, model, clusters, target_cluster):
    """
    Computes mahalanobis distance from spikes in target_cluster to all clusters in a GMM model

    Parameters
    ----------
    data : np.array,
        data used to train GMM
    model : GaussianMixture
        trained mixture model
    clusters : np.array
        maps data points to clusters
    target_cluster : int, cluster for which to compute distances

    Returns
    -------
    np.array
    """
    unique_clusters = np.unique(abs(clusters))
    out_distances = dict.fromkeys(unique_clusters)
    cluster_idx = np.where(clusters == target_cluster)[0]
    for other_cluster in unique_clusters:
        mahalanobis_dist = np.zeros((len(cluster_idx),))
        other_cluster_mean = model.means_[other_cluster, :]
        other_cluster_covar_I = linalg.inv(model.covariances_[other_cluster, :, :])
        for i, idx in enumerate(cluster_idx):
            mahalanobis_dist[i] = mahalanobis(
                data[idx, :], other_cluster_mean, other_cluster_covar_I
            )

        out_distances[other_cluster] = mahalanobis_dist

    return out_distances


def get_recording_cutoff(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec,
    **kwargs,
):
    """
    Determine the cutoff point for a recording based on the number of voltage violations.

    Parameters
    ----------
    filt_el : np.array
        Filtered electrophysiological data
    sampling_rate : int
        Sampling rate of the electrophysiological data
    voltage_cutoff : float
        Voltage cutoff for determining violations
    max_breach_rate : float
        Maximum allowed breach rate (breaches per second)
    max_secs_above_cutoff : int
        Maximum allowed number of seconds above the cutoff
    max_mean_breach_rate_persec : float
        Maximum allowed mean breach rate per second

    Returns
    -------
    int
        Recording cutoff in seconds

    """
    breach_idx = np.where(filt_el > voltage_cutoff)[0]
    breach_rate = float(len(breach_idx) * int(sampling_rate)) / len(filt_el)
    # truncate to nearest second and make 1 sec bins
    filt_el = filt_el[: int(sampling_rate) * int(len(filt_el) / sampling_rate)]
    test_el = np.reshape(filt_el, (-1, int(sampling_rate)))
    breaches_per_sec = [
        len(np.where(test_el[i] > voltage_cutoff)[0]) for i in range(len(test_el))
    ]
    breaches_per_sec = np.array(breaches_per_sec)
    secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
    if secs_above_cutoff == 0:
        mean_breach_rate_persec = 0
    else:
        mean_breach_rate_persec = np.mean(
            breaches_per_sec[np.where(breaches_per_sec > 0)[0]]
        )

    # if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el) / sampling_rate)  # cutoff in seconds
    if (
        breach_rate >= max_breach_rate
        and secs_above_cutoff >= max_secs_above_cutoff
        and mean_breach_rate_persec >= max_mean_breach_rate_persec
    ):
        # find the first 1s epoch where the number of cutoff breaches is
        # higher than the maximum allowed mean breach rate
        recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][
            0
        ]
        # cutoff is still in seconds since 1 sec bins

    return recording_cutoff


def UMAP_METRICS(waves, n_pc):
    return compute_waveform_metrics(waves, n_pc, use_umap=True)


class SpikeDetector:
    """Interface to manage spike detection and data extraction in preparation
    for GMM clustering. Intended to help create and access the neccessary
    files. If object will detect is file already exist to avoid re-creation
    unless overwrite is specified as True.
    """

    def __init__(self, file_dir, channel_name, params=None, overwrite=False, fs=None, ):
        if params is not None:
            params = params['noisy']
        # Setup paths to files and directories needed
        self.fs = fs
        self._file_dir = Path(file_dir)
        if self._file_dir.suffix == ".h5":
            self._file_dir = self._file_dir.parent
        self._channel_name = channel_name
        self._out_dir = self._file_dir / "spike_detection" / f"{channel_name}"
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self._out_dir / "data"
        self._plot_dir = self._out_dir / "plots"
        self._files = {
            "params": self._out_dir / "analysis_params" / "spike_detection_params.json",
            "spike_waveforms": self._data_dir / "spike_waveforms.npy",
            "spike_times": self._data_dir / "spike_times.npy",
            "energy": self._data_dir / "energy.npy",
            "spike_amplitudes": self._data_dir / "spike_amplitudes.npy",
            "pca_waveforms": self._data_dir / "pca_waveforms.npy",
            "slopes": self._data_dir / "spike_slopes.npy",
            "recording_cutoff": self._data_dir / "cutoff_time.txt",
            "detection_threshold": self._data_dir / "detection_threshold.txt",
        }
        for file in self._files.values():
            file.parent.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._plot_dir.mkdir(parents=True, exist_ok=True)

        self._status = dict.fromkeys(self._files.keys(), False)
        self._referenced = True

        # Delete existing data if overwrite is True
        if overwrite and self._out_dir.exists():
            shutil.rmtree(self._out_dir)

        # See what data already exists
        self._check_existing_files()

        # grab recording cutoff time if it already exists
        # cutoff should be in seconds
        self.recording_cutoff = None
        if os.path.isfile(self._files["recording_cutoff"]):
            self._status["recording_cutoff"] = True
            with open(self._files["recording_cutoff"], "r") as f:
                self.recording_cutoff = float(f.read())

        self.detection_threshold = None
        if os.path.isfile(self._files["detection_threshold"]):
            self._status["detection_threshold"] = True
            with open(self._files["detection_threshold"], "r") as f:
                self.detection_threshold = float(f.read())

        # Read in parameters
        # Parameters passed as an argument will overshadow parameters saved in file
        # Input parameters should be formatted as dataset.clustering_parameters
        if params is None and os.path.isfile(self._files["params"]):
            self.params = read_dict_from_json(self._files["params"])
        elif params is None:
            raise FileNotFoundError("params must be provided if spike_detection_params.json does not exist.")
        else:
            self.params = {}
            self.params["voltage_cutoff"] = params["data_params"][
                "V_cutoff for disconnected headstage"
            ]
            self.params["max_breach_rate"] = params["data_params"][
                "Max rate of cutoff breach per second"
            ]
            self.params["max_secs_above_cutoff"] = params["data_params"][
                "Max allowed seconds with a breach"
            ]
            self.params["max_mean_breach_rate_persec"] = params["data_params"][
                "Max allowed breaches per second"
            ]
            band_lower = params["bandpass_params"]["Lower freq cutoff"]
            band_upper = params["bandpass_params"]["Upper freq cutoff"]
            self.params["bandpass"] = [band_lower, band_upper]
            snapshot_pre = params["spike_snapshot"]["Time before spike (ms)"]
            snapshot_post = params["spike_snapshot"]["Time after spike (ms)"]
            self.params["spike_snapshot"] = [snapshot_pre, snapshot_post]
            self.params["sampling_rate"] = self.fs
            self._files["params"].parent.mkdir(parents=True, exist_ok=True)
            # Write params to json file
            write_dict_to_json(self.params, self._files["params"])
            self._status["params"] = True

    def _check_existing_files(self):
        """Checks which files already exist and updates _status so as to avoid
        re-creation later
        """
        for k, v in self._files.items():
            if os.path.isfile(v):
                self._status[k] = True
            else:
                self._status[k] = False

    def run(self):
        status = self._status
        file_dir = self._file_dir
        electrode = self._channel_name
        params = self.params

        # Check if this even needs to be run
        if all(status.values()):
            return electrode, 1, self.recording_cutoff

        print(f"Could not find referenced data for {self._channel_name}...running spike detection")
        self._referenced = False
        ref_el = h5io.get_raw_trace(file_dir, electrode)

        # Filter electrode trace
        filt_el = get_filtered_electrode(
            ref_el, freq=params["bandpass"], sampling_rate=self.fs
        )
        del ref_el
        # Get recording cutoff
        if not status["recording_cutoff"]:
            self.recording_cutoff = get_recording_cutoff(filt_el, **params)
            if not self._files["recording_cutoff"].is_file():
                self._files["recording_cutoff"].parent.mkdir(parents=True, exist_ok=True)
            with open(self._files["recording_cutoff"], "w") as f:
                f.write(str(self.recording_cutoff))

            status["recording_cutoff"] = True
            fn = os.path.join(self._plot_dir, "cutoff_time.png")
            plot_recording_cutoff(filt_el, self.fs, self.recording_cutoff, out_file=fn)

        # Truncate electrode trace, deal with early cutoff (<60s)
        if self.recording_cutoff < 60:
            print("Immediate Cutoff for electrode %i...exiting" % electrode)
            return electrode, 0, self.recording_cutoff

        filt_el = filt_el[: int(self.recording_cutoff * self.fs)]

        if not status["detection_threshold"]:
            threshold = get_detection_threshold(filt_el)
            self.detection_threshold = threshold
            with open(self._files["detection_threshold"], "w") as f:
                f.write(str(threshold))

            status["detection_threshold"] = True

        if status["spike_waveforms"] and status["spike_times"]:
            waves = np.load(self._files["spike_waveforms"])
            times = np.load(self._files["spike_times"])
        else:
            # Detect spikes and get dejittered times and waveforms
            # detect_spikes returns waveforms upsampled by 10x and times in units
            # of samples
            waves, times, threshold = detect_spikes(
                filt_el, params["spike_snapshot"], self.fs, thresh=self.detection_threshold
            )
            if waves is None:
                print("No waveforms detected on electrode %i" % electrode)
                return electrode, 0, self.recording_cutoff

            # Save waveforms and times
            np.save(self._files["spike_waveforms"], waves)
            np.save(self._files["spike_times"], times)
            status["spike_waveforms"] = True
            status["spike_times"] = True

        # Get various metrics and scale waveforms
        if not status["spike_amplitudes"]:
            amplitudes = np.min(waves)
            np.save(self._files["spike_amplitudes"], amplitudes)
            status["spike_amplitudes"] = True

        if not status["slopes"]:
            slopes = get_spike_slopes(waves)
            np.save(self._files["slopes"], slopes)
            status["slopes"] = True

        if not status["energy"]:
            energy = get_waveform_energy(waves)
            np.save(self._files["energy"], energy)
            status["energy"] = True
        else:
            energy = None

        # get pca of scaled waveforms
        if not status["pca_waveforms"]:
            scaled_waves = scale_waveforms(waves, energy=energy)
            pca_waves, explained_variance_ratio = implement_pca(scaled_waves)

            # Plot explained variance
            fn = os.path.join(self._plot_dir, "pca_variance.png")
            plot_explained_pca_variance(explained_variance_ratio, out_file=fn)

        return electrode, 1, self.recording_cutoff

    def get_spike_waveforms(self):
        """Returns spike waveforms if they have been extracted, None otherwise
        Dejittered waveforms upsampled to 10 x sampling_rate

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["spike_waveforms"]):
            return np.load(self._files["spike_waveforms"])
        else:
            return None

    def get_spike_times(self):
        """Returns spike times if they have been extracted, None otherwise
        In units of samples.

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["spike_times"]):
            return np.load(self._files["spike_times"])
        else:
            return None

    def get_energy(self):
        """Returns spike energies if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["energy"]):
            return np.load(self._files["energy"])
        else:
            return None

    def get_spike_amplitudes(self):
        """Returns spike amplitudes if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["spike_amplitudes"]):
            return np.load(self._files["spike_amplitudes"])
        else:
            return None

    def get_spike_slopes(self):
        """Returns spike slopes if they have been extracted, None otherwise

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["slopes"]):
            return np.load(self._files["slopes"])
        else:
            return None

    def get_pca_waveforms(self):
        """Returns pca of sclaed spike waveforms if they have been extracted,
        None otherwise
        Dejittered waveforms upsampled to 10 x sampling_rate, scaled to energy
        and transformed via PCA

        Returns
        -------
        numpy.array
        """
        if os.path.isfile(self._files["spike_waveforms"]):
            return np.load(self._files["spike_waveforms"])
        else:
            return None

    def get_clustering_metrics(self, n_pc=3):
        """Returns array of metrics to use for feature based clustering
        Row for each waveform with columns:
            - amplitude, energy, spike slope, PC1, PC2, etc
        """
        amplitude = self.get_spike_amplitudes()
        energy = self.get_energy()
        slopes = self.get_spike_slopes()
        pca_waves = self.get_pca_waveforms()
        out = np.vstack((amplitude, energy, slopes)).T
        out = np.hstack((out, pca_waves[:, :n_pc]))
        return out

    def __str__(self):
        out = []
        out.append("SpikeDetection\n--------------")
        out.append("Recording Directory: %s" % self._file_dir)
        out.append("Electrode: %i" % self._channel_name)
        out.append("Output Directory: %s" % self._out_dir)
        out.append("###################################\n")
        out.append("Status:")
        out.append(print_dict(self._status))
        out.append("-------------------\n")
        out.append("Parameters:")
        out.append(print_dict(self.params))
        out.append("-------------------\n")
        out.append("Data files:")
        out.append(print_dict(self._files))
        return "\n".join(out)


class ClusterGMM:
    def __init__(
        self,
        config: SortConfig,
        n_iters: int = None,
        n_restarts: int = None,
        thresh: int | float = None,
    ):
        self.cluster_params = config.get_section("cluster")
        self.n_iters = (
            n_iters if n_iters else int(self.cluster_params["max-iterations"])
        )
        self.n_restarts = (
            n_restarts if n_restarts else int(self.cluster_params["random-restarts"])
        )
        self.thresh = (
            thresh if thresh else float(self.cluster_params["convergence-criterion"])
        )

    def fit(self, data_features, n_clusters):
        """
        Perform Gaussian Mixture Model (GMM) clustering on spike waveform data.

        This method takes pre-processed waveform data and applies GMM-based clustering
        to categorize each waveform into different neuronal units. It performs multiple
        checks for the validity of the clustering solution.

        Parameters
        ----------
        data_features : array_like
            A 2D array where each row represents a waveform and each column is a feature of the waveform
            (e.g., amplitude, principal component, etc.) down each column.
        n_clusters : int
            The number of clusters to use in the GMM.

        Returns
        -------
        best_model : GaussianMixture
            The best-fitting GMM object.
        predictions : array_like
            The cluster assignments for each data point as a 1-D array.
        min_bic : float
            The minimum Bayesian information criterion (BIC) value achieved across all restarts. This value
            indicates the best-fitting model, lower number = better predictor.

        """
        min_bic = None
        best_model = None

        for i in range(self.n_restarts):  # default: 10
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                tol=self.thresh,  # default: 0.001
                random_state=i,
                max_iter=self.n_iters,  # default: 1000
            )
            model.fit(data_features)
            if model.converged_:
                new_bic = model.bic(data_features)
                if min_bic is None:
                    min_bic = model.bic(data_features)
                    best_model = model
                elif new_bic < min_bic:
                    best_model = model
                    min_bic = new_bic

        predictions = best_model.predict(data_features)
        self._model = best_model
        self._predictions = predictions
        self._bic = min_bic
        return best_model, predictions, min_bic


class CplClust(object):
    def __init__(
        self,
        rec_dirs,
        channel_number,
        out_dir=None,
        params=None,
        overwrite=False,
        no_write=False,
        n_pc=3,
        data_transform=compute_waveform_metrics,
    ):
        """Recording directories should be ordered to make spike sorting easier later on"""
        if isinstance(rec_dirs, str):
            rec_dirs = [rec_dirs]

        rec_dirs = [x[:-1] if x.endswith(os.sep) else x for x in rec_dirs]
        self.rec_dirs = rec_dirs
        self.electrode = channel_number
        self._data_transform = data_transform
        self._n_pc = n_pc
        if out_dir is None:
            if len(rec_dirs) > 1:
                top = os.path.dirname(rec_dirs[0])
                out_dir = os.path.join(top, "cpl_extract", f"{channel_number}")
            else:
                out_dir = os.path.join(rec_dirs[0], "cpl_extract", f"{channel_number}")

        if overwrite:
            shutil.rmtree(out_dir)

        # Make directories
        self.out_dir = out_dir
        self._plot_dir = os.path.join(out_dir, "plots")
        self._data_dir = os.path.join(out_dir, "clustering_results")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if not os.path.isdir(self._data_dir):
            os.mkdir(self._data_dir)

        if not os.path.isdir(self._plot_dir):
            os.mkdir(self._plot_dir)

        # Check files
        params_file = os.path.join(out_dir, "cpl_params.json")
        map_file = os.path.join(self._data_dir, "spike_id.npy")
        key_file = os.path.join(self._data_dir, "rec_key.json")
        results_file = os.path.join(self._data_dir, "clustering_results.json")
        self._files = {
            "params": params_file,
            "spike_map": map_file,
            "rec_key": key_file,
            "clustering_results": results_file,
        }
        self.params = params

        if self._rec_key is None and not no_write:
            # Create new rec key
            rec_key = {x: y for x, y in enumerate(self.rec_dirs)}
            self._rec_key = rec_key
            write_dict_to_json(rec_key, self._files["rec_key"])
        elif self._rec_key is None:
            ValueError("Existing rec_key not found and no_write is enabled")

        # Check to see if spike detection is already completed on all recording directories
        spike_check = self._check_spike_detection()
        if not all(spike_check):
            invalid = [rec_dirs[i] for i, x in enumerate(spike_check) if x == False]
            error_str = "\n\t".join(invalid)
            raise ValueError("Spike detection has not been run on:\n\t%s" % error_str)

    def run(self, n_pc=None, overwrite=False):
        if self.clustered and not overwrite:
            return True

        if n_pc is None:
            n_pc = self._n_pc

        GMM = ClusterGMM(
            self.params["max_iterations"],
            self.params["num_restarts"],
            self.params["threshold"],
        )

        # Collect data from all recordings
        waveforms, spike_times, spike_map, self.fs, offsets = self.get_spike_data()

        # Save array to map spikes and predictions back to original recordings
        np.save(self._files["spike_map"], spike_map)

        data, data_columns = self._data_transform(waveforms, n_pc)
        amplitudes = np.min(waveforms)

        # Run GMM for each number of clusters from 2 to max_clusters
        tested_clusters = np.arange(2, self.params["max_clusters"] + 1)
        clust_results = pd.DataFrame(
            columns=["clusters", "converged", "BIC", "spikes_per_cluster"],
            index=tested_clusters,
        )
        for n_clust in tested_clusters:
            data_dir = os.path.join(self._data_dir, "%i_clusters" % n_clust)
            plot_dir = os.path.join(self._plot_dir, "%i_clusters" % n_clust)
            wave_plot_dir = os.path.join(
                self._plot_dir, "%i_clusters_waveforms_ISIs" % n_clust
            )
            bic_file = os.path.join(data_dir, "bic.npy")
            pred_file = os.path.join(data_dir, "predictions.npy")

            if os.path.isfile(bic_file) and os.path.isfile(pred_file) and not overwrite:
                bic = np.load(bic_file)
                predictions = np.load(pred_file)
                spikes_per_clust = [
                    len(np.where(predictions == c)[0]) for c in np.unique(predictions)
                ]
                clust_results.loc[n_clust] = [n_clust, True, bic, spikes_per_clust]
                continue

            if not os.path.isdir(wave_plot_dir):
                os.makedirs(wave_plot_dir)

            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)

            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            model, predictions, bic = GMM.fit(data, n_clust)
            if model is None:
                clust_results.loc[n_clust] = [n_clust, bic, False, [0]]
                # Nothing converged
                continue

            # Go through each cluster and throw out any spikes too far from the
            # mean
            spikes_per_clust = []
            for c in range(n_clust):
                idx = np.where(predictions == c)[0]
                mean_amp = np.mean(amplitudes[idx])
                sd_amp = np.std(amplitudes[idx])
                cutoff_amp = mean_amp - (sd_amp * self.params["wf_amplitude_sd_cutoff"])
                rejected_idx = np.array([i for i in idx if amplitudes[i] <= cutoff_amp])
                if len(rejected_idx) > 0:
                    predictions[rejected_idx] = -1

                idx = np.where(predictions == c)[0]
                spikes_per_clust.append(len(idx))

                if len(idx) == 0:
                    continue

                # Plot waveforms and ISIs of cluster
                ISIs, violations_1ms, violations_2ms = get_ISI_and_violations(
                    spike_times[idx], self.fs, spike_map[idx]
                )
                cluster_waves = waveforms[idx]
                cluster_times = spike_times[idx]
                isi_fn = os.path.join(wave_plot_dir, "Cluster%i_ISI.png" % c)
                wave_fn = os.path.join(wave_plot_dir, "Cluster%i_waveforms.png" % c)
                title_str = (
                    "Cluster%i\nviolations_1ms = %i, "
                    "violations_2ms = %i\n"
                    "Number of waveforms = %i"
                    % (c, violations_1ms, violations_2ms, len(idx))
                )
                plot_waveforms(cluster_waves, title=title_str, save_file=wave_fn)
                if len(ISIs) > 0:
                    plot_ISIs(ISIs, total_spikes=len(idx), save_file=isi_fn)

            clust_results.loc[n_clust] = [n_clust, True, bic, spikes_per_clust]

            # Plot feature pairs
            feature_pairs = itertools.combinations(list(range(data.shape[1])), 2)
            for f1, f2 in feature_pairs:
                fn = "%sVS%s.png" % (data_columns[f1], data_columns[f2])
                fn = os.path.join(plot_dir, fn)
                plot_cluster_features(
                    data[:, [f1, f2]],
                    predictions,
                    x_label=data_columns[f1],
                    y_label=data_columns[f2],
                    save_file=fn,
                )

            # For each cluster plot mahanalobis distances to all other clusters
            for c in range(n_clust):
                distances = get_mahalanobis_distances_to_cluster(
                    data, model, predictions, c
                )
                fn = os.path.join(plot_dir, "Mahalanobis_cluster%i.png" % c)
                title = "Mahalanobis distance of Cluster %i from all other clusters" % c
                plot_mahalanobis_to_cluster(distances, title=title, save_file=fn)

            # Save data
            np.save(bic_file, bic)
            np.save(pred_file, predictions)

        # Save results table
        self.results = clust_results
        write_pandas_to_table(
            clust_results, self._files["clustering_results"], overwrite=True
        )
        self.clustered = True
        return True

    def get_spike_data(self):
        # Collect data from all recordings
        tmp_waves = []
        tmp_times = []
        tmp_id = []
        fs = dict.fromkeys(self._rec_key.keys())
        offsets = dict.fromkeys(self._rec_key.keys())
        offset = 0
        for i in sorted(self._rec_key.keys()):
            rec = self._rec_key[i]
            spike_detect = SpikeDetector(rec, self.electrode)
            t = spike_detect.get_spike_times()
            fs[i] = spike_detect.params["sampling_rate"]
            if t is None:
                offsets[i] = int(offset)
                offset = offset + 3 * fs[i]
                continue

            tmp_waves.append(spike_detect.get_spike_waveforms())
            tmp_times.append(t)
            tmp_id.append(np.ones((t.shape[0],)) * i)
            offsets[i] = int(offset)
            offset = offset + max(t) + 3 * fs[i]

        waveforms = np.vstack(tmp_waves)
        spike_times = np.hstack(tmp_times)
        spike_map = np.hstack(tmp_id)

        # Double check that spike_map matches up with existing spike_map
        if os.path.isfile(self._files["spike_map"]):
            orig_map = np.load(self._files["spike_map"])
            if len(orig_map) != len(spike_map):
                raise ValueError(
                    "Spike detection has changed, please re-cluster with overwrite=True"
                )

        return waveforms, spike_times, spike_map, fs, offsets
    def _load_existsing_data(self):
        params = self.params
        file_check = self._check_existing_files()

        # Check params files and create if new params are passed
        if file_check["params"]:
            self.params = read_dict_from_json(self._files["params"])

        # Make new params or overwrite existing with passed params
        if params is None and not file_check["params"]:
            raise ValueError(
                (
                    "Params file does not exists at %s. Must provide"
                    " clustering parameters."
                )
                % self._files["params"]
            )
        elif params is not None:
            self.params["max_clusters"] = params["clustering_params"][
                "Max Number of Clusters"
            ]
            self.params["max_iterations"] = params["clustering_params"][
                "Max Number of Iterations"
            ]
            self.params["threshold"] = params["clustering_params"][
                "Convergence Criterion"
            ]
            self.params["num_restarts"] = params["clustering_params"][
                "GMM random restarts"
            ]
            self.params["wf_amplitude_sd_cutoff"] = params["data_params"][
                "Intra-cluster waveform amp SD cutoff"
            ]
            write_dict_to_json(self.params, self._files["params"])

        # Deal with existing rec key
        if file_check["rec_key"]:
            rec_dirs = self.rec_dirs
            rec_key = read_dict_from_json(self._files["rec_key"])
            rec_key = {int(x): y for x, y in rec_key.items()}
            if len(rec_key) != len(rec_dirs):
                raise ValueError("Rec key does not match rec dirs")

            # Correct rec key in case rec_dir roots have changed
            for rd in rec_dirs:
                rn = os.path.basename(rd)
                dn = os.path.dirname(rd)
                kd = [(x, y) for x, y in rec_key.items() if rn in y]
                if len(kd) == 0:
                    raise ValueError("%s not found in rec_key" % rn)

                kd = kd[0]
                if kd[1] != rd:
                    rec_key[kd[0]] = rd

            inverted = {v: k for k, v in rec_key.items()}
            self.rec_dirs = sorted(self.rec_dirs, key=lambda i: inverted[i])
            self._rec_key = rec_key
        else:
            self._rec_key = None

        # Check is clustering has already been done, load results
        if file_check["clustering_results"]:
            self.results = read_pandas_from_table(self._files["clustering_results"])
            expected_results = np.arange(2, self.params["max_clusters"] + 1)
            if not all([x in self.results["clusters"] for x in expected_results]):
                self.clustered = False
            else:
                self.clustered = True

        else:
            self.results = None
            self.clustered = False

    def _check_existing_files(self):
        out = dict.fromkeys(self._files.keys(), False)
        for k, v in self._files.items():
            if os.path.isfile(v):
                out[k] = True

        return out

    def _check_spike_detection(self):
        out = []
        for rec in self.rec_dirs:
            try:
                spike_detect = SpikeDetector(rec, self.electrode)
                if all(spike_detect._status):
                    out.append(True)
                else:
                    out.append(False)

            except FileNotFoundError:
                out.append(False)

        return out
