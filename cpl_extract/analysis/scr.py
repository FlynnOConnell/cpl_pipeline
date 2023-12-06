from __future__ import annotations

from itertools import combinations

import mne_connectivity

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from mne.time_frequency import psd_array_welch
from mne_connectivity import spectral_connectivity_epochs
from scipy.stats import zscore, sem, ttest_rel
from sklearn.manifold import spectral_embedding  # noqa

from cpl_extract.analysis.stats import extract_file_info
from cpl_extract.logs import logger
from cpl_extract.spk_io.utils import read_npz_as_dict
from cpl_extract.utils import extract_common_key
from cpl_extract.viz import helpers
from cpl_extract.viz import plots

sns.set_style("darkgrid")

logger.setLevel("INFO")


def _validate_events(event_arr, event_id):
    """
    Removes events that occur less than 3 times.

    Parameters:
    -----------
    event_arr (np.array): Array of events
    event_id (dict): Dictionary of event ids

    Returns:
    --------
    valid_times (np.array): Array of valid events
    valid_event_id (dict): Dictionary of valid event ids
    """
    assert event_arr.shape[1] == 3, "Event array should have 3 columns"

    valid_event_id = {}
    for event_name, event_val in event_id.items():
        occurrences = sum(event_arr[:, -1] == event_val)
        if occurrences > 2:
            valid_event_id[event_name] = event_val

    # Remove invalid events from times array
    valid_values = list(valid_event_id.values())
    valid_times = np.array([row for row in event_arr if row[-1] in valid_values])

    return valid_times, valid_event_id


def get_lowest_variance_window(arr, sfreq, window_length_sec=10):

    arr = arr.get_data()
    window_length = int(window_length_sec * sfreq)  # Convert seconds to samples
    step_size = window_length // 2  # 50% overlap
    lowest_variance = float("inf")
    best_window = (0, window_length)
    best_window_data = None

    # Iterate over possible windows
    for start in range(0, arr.shape[-1] - window_length + 1, step_size):

        end = start + window_length
        window = arr[:, start:end]
        window_variance = np.var(window)

        if window_variance < lowest_variance:
            lowest_variance = window_variance
            best_window = (start, end)
            best_window_data = window

    if best_window_data is not None:
        return best_window_data


def get_master_df():
    data_path = Path().home() / "data" / ".cache"
    animals = sorted(
        [x for x in data_path.iterdir() if x.is_dir() and not x.name.startswith(".")]
    )
    animals = animals[:-1]

    master = pd.DataFrame()
    for animal_path in animals:

        cache_animal_path = data_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(list(animal_path.glob("*_raw*.fif")), key=extract_common_key)
        event_files = sorted(
            list(animal_path.glob("*_eve*.fif")), key=extract_common_key
        )
        event_id = sorted(list(animal_path.glob("*_id_*.npz")), key=extract_common_key)
        assert len(raw_files) == len(event_files) == len(event_id)
        sfreq = raw_files

        metadata = [extract_file_info(raw_file.name) for raw_file in raw_files]
        raw_data = [mne.io.read_raw_fif(raw_file) for raw_file in raw_files]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        event_id_dicts = [read_npz_as_dict(id_file) for id_file in event_id]

        first_event_times = [event[0][0] for event in event_data]
        first_event_times = [np.array([time]) for time in first_event_times]
        baseline_windows = [
            raw_data[i].copy().crop(tmin=0, tmax=time[0] / 2000)
            for i, time in enumerate(first_event_times)
        ]
        baseline_segment = [
            get_lowest_variance_window(window, raw_data[i].info["sfreq"])
            for i, window in enumerate(baseline_windows)
        ]
        # map each baseline channel to its baseline segment

        ev_id_dict = [
            {k: v for k, v in zip(ev_id_dict["keys"], ev_id_dict["values"])}
            for ev_id_dict in event_id_dicts
        ]

        ev_start_holder = []
        ev_end_holder = []
        for (
            arr
        ) in (
            event_data
        ):  # separate the start and end events to work with mne event structures [start, 0, id]
            start_events = np.column_stack(
                [arr[:, 0], np.zeros(arr.shape[0]), arr[:, 2]]
            ).astype(int)
            end_events = np.column_stack(
                [arr[:, 1], np.zeros(arr.shape[0]), arr[:, 2]]
            ).astype(int)

            ev_start_holder.append(start_events)
            ev_end_holder.append(end_events)

        all_data_dict = {  # each v is a list for each file, sorted to be the smae order
            "raw": raw_data,
            "start": ev_start_holder,
            "end": ev_end_holder,
            "event_id": ev_id_dict,
            "baseline": baseline_segment,
        }

        metadata_df = pd.concat(metadata).reset_index(drop=True)
        all_data_df = pd.DataFrame(all_data_dict)
        all_data_df = pd.concat([metadata_df, all_data_df], axis=1)
        master = pd.concat([master, all_data_df], axis=0)
    return master


def preprocess_raw(raw_signal, fmin=1, fmax=100):
    raw: mne.io.RawArray = raw_signal.copy()
    raw.load_data()
    raw.filter(fmin, fmax, fir_design="firwin")
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    raw.set_eeg_reference(ref_channels=["Ref"])
    raw.drop_channels(["Ref"])
    return raw


def update_df_with_epochs(
    df,
    fmin,
    fmax,
    tmin,
    tmax,
    evt_row="end",
):
    epochs_holder, con_holder = [], []
    for idx, row in df.iterrows():
        epoch, con_data = process_epoch(
            row["raw"],
            row[evt_row],
            row["event_id"],
            row["baseline"],
            fmin,
            fmax,
            tmin,
            tmax,
        )
        epochs_holder.append(epoch)

        con_holder.append(con_data)
    df["epoch"] = epochs_holder
    df["con_arr"] = con_holder
    return df


def process_epoch(raw_arr, events, event_id, baseline, fmin, fmax, tmin, tmax):

    raw = preprocess_raw(raw_arr, fmin, fmax)
    means = np.mean(baseline, axis=1)
    stds = np.std(baseline, axis=1)
    means = means[:-1]
    stds = stds[:-1]

    valid_events, valid_event_id = _validate_events(events, event_id)

    raw = raw.filter(fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1)
    epochs = mne.Epochs(
        raw,
        valid_events,
        valid_event_id,
        tmin,
        tmax,
        baseline=None,
        event_repeated="drop",
        preload=True,
    )

    if epochs is None:
        raise ValueError("Epoch is None")

    for ch in epochs.ch_names:
        ch_idx = epochs.ch_names.index(ch)
        epochs._data[:, ch_idx, :] = (
            epochs._data[:, ch_idx, :] - means[ch_idx]
        ) / stds[ch_idx]

    freqs_arange = np.arange(fmin, fmax)

    con = spectral_connectivity_epochs(
        epochs,
        indices=(np.array([0]), np.array([1])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs_arange,
        cwt_n_cycles=freqs_arange / 2,
        sfreq=epochs.info["sfreq"],
        n_jobs=1,
    )
    return epochs, con


def main(use_parallel=True):
    helpers.update_rcparams()
    data = get_master_df().reset_index(drop=True)
    return data


def optimize_psd_params(sfreq, time_series_length, desired_resolution=None):
    """
    Optimize PSD calculation parameters.

    Parameters:
    - sfreq: Sampling frequency
    - time_series_length: Length of the time series data
    - desired_resolution: Desired frequency resolution (optional)

    Returns:
    - n_fft: Optimal FFT length
    - n_per_seg: Optimal number of samples per segment
    - n_overlap: Optimal number of overlap samples
    """

    # Choose n_fft as the next power of 2 greater than time_series_length for computational efficiency
    n_fft = 2 ** np.ceil(np.log2(time_series_length))
    n_fft = int(min(n_fft, time_series_length))

    if desired_resolution:
        n_fft = int(sfreq / desired_resolution)

    n_per_seg = n_fft  # You can make this smaller to increase the number of segments averaged over
    n_overlap = int(n_per_seg / 2)

    return n_fft, n_per_seg, n_overlap


def compute_coherence_base(epoch_obj, idxs, freqs, sfreq):
    if hasattr(epoch_obj, "info"):
        assert sfreq == epoch_obj.info["sfreq"]
        epoch_obj = epoch_obj.get_data()

    epoch_connectivity = spectral_connectivity_epochs(
        epoch_obj,
        indices=(np.array([idxs[0]]), np.array([idxs[1]])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=freqs / 2,
        sfreq=sfreq,
        n_jobs=1,
    )
    return epoch_connectivity


def compute_power(epoch_obj, sfreq=None, n_fft=None, n_per_seg=None, n_overlap=None):
    fs, arr = None, None

    if not hasattr(epoch_obj, "info") and sfreq is None:
        raise ValueError(
            "Must provide sampling frequency if epoch_obj is not an Epochs object"
        )

    if not hasattr(epoch_obj, "info") and sfreq is not None:
        fs = sfreq
        arr = epoch_obj

    if hasattr(epoch_obj, "info") and sfreq is not None:
        fs = epoch_obj.info["sfreq"]
        arr = epoch_obj.get_data()

    if hasattr(epoch_obj, "info") and sfreq is None:
        fs = epoch_obj.info["sfreq"]
        arr = epoch_obj.get_data()

    # Compute PSD
    power = psd_array_welch(
        arr, fs, n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, n_jobs=1
    )
    return power


def set_event_id(epoch):

    epoch.event_id = all_event_keys
    return epoch


def determine_group(chan_pair_names):
    """
    Determines whether a pair of channels is within or between brain regions.

    Parameters
    ----------
    chan_pair_names : tuple
        Tuple of channel names.

    Returns
    -------
    str
        'within' or 'between'

    """
    if any(all(x in chan_pair_names for x in w) for w in brain_regions_within):
        return "within"
    elif any(all(x in chan_pair_names for x in b) for b in brain_regions_between):
        return "between"
    else:
        raise ValueError(f"Invalid pair name: {chan_pair_names}")


if __name__ == "__main__":

    data = main(use_parallel=False)
    epoch_tmin, epoch_tmax = -0.5, 0.5
    pre_fmin, pre_fmax = 13, 30
    c1 = ["context", "nocontext"]
    c2 = ["between", "within"]

    brain_regions_within = [("LFP1_vHp", "LFP2_vHp"), ("LFP3_AON", "LFP4_AON")]
    brain_regions_between = [
        ("LFP1_vHp", "LFP3_AON"),
        ("LFP1_vHp", "LFP4_AON"),
        ("LFP2_vHp", "LFP3_AON"),
        ("LFP2_vHp", "LFP4_AON"),
    ]
    brain_regions_all = ["LFP1_vHp", "LFP2_vHp", "LFP3_AON", "LFP4_AON"]

    all_animals = data["Animal"].unique()
    all_event_keys = {"b_1": 1, "b_0": 2, "w_1": 3, "w_0": 4, "x_1": 5, "x_0": 6}

    lfp_bands = {"beta": (12, 30), "gamma": (30, 80), "theta": (4, 12)}
    chan_indices = np.array(list(combinations(range(4), 2)))

    animal_df = data[data["Animal"] == all_animals[0]]
    condition = c1[0]

    df = update_df_with_epochs(
        animal_df[animal_df["Context"] == condition],
        pre_fmin,
        pre_fmax,
        epoch_tmin,
        epoch_tmax,
        "end",
    )
    df = df.iloc[0]
    baseline = df["baseline"].tolist()
    baseline = np.array([b[:-1] for b in baseline])  # remove the last channel (ref)

    epochs = list(map(lambda e: set_event_id(e), df["epoch"].tolist()))

    sfreq = epochs[0].info["sfreq"]
    numt = epochs[0].times.size
    params = optimize_psd_params(sfreq, numt)

    power_epochs = [compute_power(x, sfreq, *params) for x in epochs]
    power_epochs_data = [x[0] for x in power_epochs]

    power_baseline = [compute_power(bl, sfreq, *params) for bl in baseline]
    power_baseline_data = [bl[0] for bl in power_baseline]
    power_freqs = power_baseline_data[0][0]

    # Calculate the mean and SEM of baseline power across channels (if needed)
    mean_bl_power = np.mean([bl.mean(axis=0) for bl in power_baseline_data], axis=0)
    sem_bl_power = sem([bl.mean(axis=0) for bl in power_baseline_data], axis=0)

    # Initialize list to store percent change for each epoch
    pct_change_power = []

    # Loop through each epoch's power data
    for epoch_power in power_epochs_data:
        # Mean power across channels for this epoch
        mean_epoch_power = epoch_power.mean(axis=0)
        # Calculate percent change
        percent_change_power = (
            (mean_epoch_power - mean_bl_power) / mean_bl_power
        ) * 100
        pct_change_power.append(percent_change_power)

    # Convert to array for easier manipulation
    pct_change_power = np.array(pct_change_power)

    # Calculate the mean and SEM across epochs
    mean_pct_change = pct_change_power.mean(axis=0)
    mean_c = mean_pct_change.mean(axis=0)
    sem_pct_change = sem(pct_change_power, axis=0)
    sem_c = sem_pct_change.mean(axis=0)

    # After calculating percent_change_power for each epoch
    percent_change_power = np.clip(mean_c, 0, 100)

    sorted_indices = np.argsort(power_freqs)
    sorted_freqs = power_freqs[sorted_indices]

    # Sort the percent change data according to the sorted frequency indices
    sorted_pct_change = pct_change_power[:, :, sorted_indices]

    # Calculate the mean percent change across channels for each session, now with sorted data
    mean_sorted_pct_change_across_channels = sorted_pct_change.mean(axis=1)

    # Plotting individual sessions to check the patterns
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 30), sharex=True)

    for i, ax in enumerate(axes.flatten()[:5]):  # plot first 5 for example
        if i < len(mean_sorted_pct_change_across_channels):
            ax.plot(
                sorted_freqs,
                mean_sorted_pct_change_across_channels[i],
                label=f"Session {i}",
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Change (%)")
            ax.legend()
    plt.show()

    # for ch_pair in combinations(range(len(brain_regions_all)), 2):
    #
    #     pair_name = (brain_regions_all[ch_pair[0]], brain_regions_all[ch_pair[1]])
    #     group = determine_group(pair_name)
    #     if group == 'within':
    #         continue
    #
    #     all_animals_data[animal][condition][group] = {}
    #
    #     sfreq = epochs[0].info['sfreq']
    #     numt = epochs[0].times.size
    #     params = optimize_psd_params(sfreq, numt)
    #
    #     welch_epochs = [scipy.signal.welch(epoch, epoch.info['sfreq'], nperseg=1000, axis=-1) for epoch in epochs]
    #     power_epochs = [compute_power(x, sfreq, *params)for x in epochs]
    #     power_epochs_data = [x[0] for x in power_epochs]
    #     logger.debug(f"Power epochs shape: {power_epochs_data[0].shape}")
    #
    #     power_baseline = [compute_power(bl, sfreq, *params) for bl in baseline]
    #     power_baseline_data = [bl[0] for bl in power_baseline]
    #     logger.debug(f"Power baseline shape: {power_baseline_data[0].shape}")
    #     power_freqs = power_baseline_data[0][0]
    #     logger.debug(f"Power freqs: {power_freqs}")
    #
    #     # Calculate the mean and SEM of baseline power across channels (if needed)
    #     mean_bl_power = np.mean([bl.mean(axis=0) for bl in power_baseline_data], axis=0)
    #     sem_bl_power = sem([bl.mean(axis=0) for bl in power_baseline_data], axis=0)
    #
    #     # Initialize list to store percent change for each epoch
    #     pct_change_power = []
    #
    #     # Loop through each epoch's power data
    #     for epoch_power in power_epochs_data:
    #         # Mean power across channels for this epoch
    #         mean_epoch_power = epoch_power.mean(axis=0)
    #         # Calculate percent change
    #         percent_change_power = ((mean_epoch_power - mean_bl_power) / mean_bl_power) * 100
    #         pct_change_power.append(percent_change_power)
    #
    #     # Convert to array for easier manipulation
    #     pct_change_power = np.array(pct_change_power)
    #
    #     # Calculate the mean and SEM across epochs
    #     mean_pct_change = pct_change_power.mean(axis=0)
    #     mean_c = mean_pct_change.mean(axis=0)
    #     sem_pct_change = sem(pct_change_power, axis=0)
    #     sem_c = sem_pct_change.mean(axis=0)
    #
    #     # After calculating percent_change_power for each epoch
    #     percent_change_power = np.clip(mean_c, 0, 100)
    #
    #     # Plotting
    #     # fig, ax = plt.subplots(figsize=(10, 6))
    #     # ax.plot(power_freqs, mean_c, label='Mean Percent Change')
    #     # ax.fill_between(power_freqs, mean_c - sem_c, mean_c + sem_c, color='k', alpha=0.2)
    #     # ax.set_xlabel('Frequency (Hz)')
    #     # ax.set_xticks(power_freqs)
    #     # ax.set_ylabel('Percent Change in Power Spectral Density (%)')
    #     # ax.legend()
    #     # plt.show()
    #
    #     sorted_indices = np.argsort(power_freqs)
    #     sorted_freqs = power_freqs[sorted_indices]
    #
    #     # Sort the percent change data according to the sorted frequency indices
    #     sorted_pct_change = pct_change_power[:, :, sorted_indices]
    #
    #     # Calculate the mean percent change across channels for each session, now with sorted data
    #     mean_sorted_pct_change_across_channels = sorted_pct_change.mean(axis=1)
    #
    #     # Plotting individual sessions to check the patterns
    #     fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 30), sharex=True)
    #     plt.title
    #     for i, ax in enumerate(axes.flatten()[:5]):  # plot first 5 for example
    #         if i < len(mean_sorted_pct_change_across_channels):
    #             ax.plot(sorted_freqs, mean_sorted_pct_change_across_channels[i], label=f'Session {i}')
    #             ax.set_xlabel('Frequency (Hz)')
    #             ax.set_ylabel('Power Change (%)')
    #             ax.legend()
    #     plt.show()

    #
    # # Calculate the percentage change in power
    # pct_change_power = []
    # for i, (baseline_power, epoch_power) in enumerate(zip(power_baseline_data, power_epochs_data)):
    #
    #     # Reshape mean_bl_power to broadcast across epochs and frequency bins
    #     mean_bl_power = np.mean(baseline_power, axis=0)
    #
    #     percent_change_power = (epoch_power - mean_bl_power) / mean_bl_power * 100
    #     pct_change_power.append(percent_change_power)
    #     sem_bl_power = sem(baseline_power, axis=1)
    #
    #     # Calculate the mean and SEM for the percentage change
    #     mean_change_per_channel = np.mean(percent_change_power, axis=0)  # per cell avg
    #     sem_change_per_channel = sem(percent_change_power, axis=0)
    #
    #     mean_change = np.mean(mean_change_per_channel, axis=0)  # avg across cells
    #     sem_change = sem(mean_change_per_channel, axis=0)
    #
    #     # Plot the percentage change with the SEM as the shaded area
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.plot(power_freqs, mean_change, color="k")
    #
    #     plt.xlabel("Frequency (Hz)")
    #     plt.ylabel("Percent change in power")
    #     plt.title(f"Percent change in power for {pair_name} in {condition} condition")
    #     plt.show()
