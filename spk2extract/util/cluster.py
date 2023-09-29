from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numba import jit
from numba.typed import List
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logpath = Path().home() / 'autosort' / "cluster.log"


def filter_signal(sig, sampling_rate: int | float, freq=(300, 6000), ):
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
        signal: np.ndarray, sampling_rate: int | float, spike_snapshot: tuple = (0.2, 0.6), STD=2.0, cutoff_std=10.0
):
    """
    Extract individual spike waveforms from the filtered electrode signal.

    Parameters
    ----------
    signal : array-like
        The (already bandpass-filtered) electrode signal as a 1-D array.
    sampling_rate : float
        The sampling rate of the signal in Hz. Default is 20000.0 Hz.
    spike_snapshot : tuple of float, optional
        The time range (in milliseconds) around each spike to extract, given as (pre_spike_time, post_spike_time).
        Default is (0.2, 0.6).
    STD : float, optional
        The number of standard deviations to use for the spike detection threshold. Default is 2.0.
    cutoff_std : float, optional
        The cutoff threshold, in units of the standard deviation, for rejecting spikes. Default is 10.0.

    Returns
    -------
    slices : list of array-like
        List of extracted spike waveforms, each as a 1-D array.
    spike_times : list of int
        List of indices indicating the positions of the extracted spikes in the input array.
    """
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
            signal[pos[changes[i]: changes[i + 1]]]
            == np.min(signal[pos[changes[i]: changes[i + 1]]])
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
                        - int((spike_snapshot[0] + 0.1) * (sampling_rate / 1000.0)): pos[
                                                                                         minimum[0] + changes[i]
                                                                                         ]
                                                                                     + int(
                            (spike_snapshot[1] + 0.1) * (sampling_rate / 1000.0))
                        ]
            if ~np.any(np.absolute(tempslice) > (th * cutoff_std) / STD):
                slices.append(tempslice)
                spike_times.append(pos[minimum[0] + changes[i]])

    return slices, spike_times


def dejitter(slices, spike_times, sampling_rate, spike_snapshot=(0.2, 0.6), ):
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

    slices_dejittered = List()
    spike_times_dejittered = List()
    for i in range(len(slices)):
        f = interp1d(x, slices[i])
        # 10-fold interpolated spike
        ynew = f(xnew)
        minimum = np.where(ynew == np.min(ynew))[0][0]
        # Only accept spikes if the interpolated minimum has shifted by less than 1/10th of a ms (4 samples for a
        # 40kHz recording, 40 samples after interpolation) If minimum hasn't shifted at all, then minimum - 5ms
        # should be equal to zero (because we sliced 5 ms before the minimum in extract_waveforms())
        if np.abs(
                minimum - int((spike_snapshot[0] + 0.1) * (sampling_rate / 100.0))
        ) < int(10.0 * (sampling_rate / 10000.0)):
            slices_dejittered.append(ynew[minimum - before * 10: minimum + after * 10])
            spike_times_dejittered.append(spike_times[i])
    return slices_dejittered, spike_times_dejittered


def scale_waveforms(slices_dejittered):
    """
    Scale the extracted spike waveforms by their energy.

    Parameters
    ----------
    slices_dejittered : array-like
        The dejittered spike waveforms as a 2-D array.

    Returns
    -------
    scaled_slices : array-like
        The scaled spike waveforms as a 2-D array.
    energy : array-like
        The energy of each spike waveform as a 1-D array.
    """
    energy = np.sqrt(np.sum(slices_dejittered ** 2, axis=1)) / len(slices_dejittered[0])
    scaled_slices = np.zeros((len(slices_dejittered), len(slices_dejittered[0])))
    for i in range(len(slices_dejittered)):
        scaled_slices[i] = slices_dejittered[i] / energy[i]

    return scaled_slices, energy


def implement_pca(scaled_slices):
    """
    Perform PCA (Principal Component Analysis) on the scaled spike waveforms.

    Parameters
    ----------
    scaled_slices : array-like
        The scaled spike waveforms as a 2-D array.

    Returns
    -------
    pca_slices : array-like
        The principal components of the spike waveforms as a 2-D array.
    explained_variance_ratio : array-like
        The proportion of variance explained by each principal component as a 1-D array.
    """
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)
    return pca_slices, pca.explained_variance_ratio_


def cluster_gmm(data, n_clusters, n_iter, restarts, threshold):
    """
    Cluster the data using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : array-like
        The input data to be clustered, generally the output of a PCA, as a 2-D array.
    n_clusters : int
        The number of clusters to use in the GMM.
    n_iter : int
        The maximum number of iterations to perform in the GMM.
    restarts : int
        The number of times to restart the GMM with different initializations.
    threshold : float
        The convergence threshold for the GMM.

    Returns
    -------
    best_fit_gmm : object
        The best-fitting GMM object.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.
    min_bayesian : float
        The minimum Bayesian information criterion (BIC) value achieved across all restarts.
    """

    g = []
    bayesian = []

    # Run the GMM
    try:
        for i in range(restarts):
            g.append(
                GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    tol=threshold,
                    random_state=i,
                    max_iter=n_iter,
                )
            )
            g[-1].fit(data)
            if g[-1].converged_:
                bayesian.append(g[-1].bic(data))
            else:
                del g[-1]

        # print len(akaike)
        bayesian = np.array(bayesian)
        best_fit = np.where(bayesian == np.min(bayesian))[0][0]

        predictions = g[best_fit].predict(data)
        return g[best_fit], predictions, np.min(bayesian)
    except Exception as e:
        logger.warning(f"Error in clusterGMM: {e}", exc_info=True)


def get_lratios(data, predictions):
    """
    Calculate L-ratios for each cluster, a measure of cluster quality.

    Parameters
    ----------
    data : array-like
        The input data, generally the output of a PCA, as a 2-D array.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.

    Returns
    -------
    Lrats : dict
        A dictionary with cluster labels as keys and the corresponding L-ratio values as values.
    """
    Lrats = {}
    df = np.shape(data)[1]
    for ref_cluster in np.sort(np.unique(predictions)):
        if ref_cluster < 0:
            continue
        ref_mean = np.mean(data[np.where(predictions == ref_cluster)], axis=0)
        ref_covar_I = linalg.inv(
            np.cov(data[np.where(predictions == ref_cluster)], rowvar=False)
        )
        Ls = [
            1 - chi2.cdf((mahalanobis(data[point, :], ref_mean, ref_covar_I)) ** 2, df)
            for point in np.where(predictions[:] != ref_cluster)[0]
        ]
        Lratio = sum(Ls) / len(np.where(predictions == ref_cluster)[0])
        Lrats[ref_cluster] = Lratio
    return Lrats
