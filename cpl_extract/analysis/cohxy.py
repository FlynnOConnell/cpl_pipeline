from __future__ import annotations

from itertools import combinations

import mne_connectivity

from pathlib import Path

import mne
import numpy as np
import pandas as pd
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

import matplotlib

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


def get_lowest_variance_window(arr, sfreq, window_length_sec=5):

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
        first_event_times = [event[0][0] for event in event_data]

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
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    if "Ref" in raw.ch_names:
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
        try:
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
        except Exception as e:
            print(e)
            epochs_holder.append(None)

    df["epoch"] = epochs_holder
    df["con_arr"] = con_holder
    return df


def process_epoch(raw_arr, events, event_id, baseline, fmin, fmax, tmin, tmax):
    try:
        raw = preprocess_raw(raw_arr, fmin, fmax)
        means = np.mean(baseline, axis=1)
        stds = np.std(baseline, axis=1)
        means = means[:-1]
        stds = stds[:-1]

        valid_events, valid_event_id = _validate_events(events, event_id)
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
    except Exception as e:
        print(e)
        return None, None


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


def _normalize_to_baseline(epoch, baseline):
    epoch_data = epoch.get_data()
    normalized_data = np.empty_like(epoch_data)

    for ch in range(epoch_data.shape[1]):
        for ep in range(epoch_data.shape[0]):
            baseline_mean = baseline[ch, :].mean()
            normalized_data[ep, ch, :] = (
                (epoch_data[ep, ch, :] - baseline_mean) / baseline_mean * 100
            )

    normalized_epochs = mne.EpochsArray(
        normalized_data, epoch.info, events=epoch.events, tmin=epoch.tmin
    )

    return normalized_epochs


def average_coherence_change(baseline_coherence, epochs_coherence):
    percent_changes = []
    for baseline_coh, epoch_coh in zip(baseline_coherence, epochs_coherence):
        mean_baseline_coh = np.mean(baseline_coh, axis=1, keepdims=True)
        mean_baseline_coh = mean_baseline_coh[4:, :]
        percent_change = (epoch_coh - mean_baseline_coh) / mean_baseline_coh * 100
        percent_changes.append(percent_change)
    return np.mean(percent_changes, axis=0), sem(percent_changes, axis=0)


if __name__ == "__main__":

    data = main(use_parallel=False)
    epoch_tmin, epoch_tmax = -0.5, 0.5
    pre_fmin, pre_fmax = 1, 100
    c1 = ["context", "nocontext"]

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
    sfreq = 1000

    all_animals_data = {}
    animal_list = ["dk1", "dk5"]

    for animal in all_animals:

        all_animals_data[animal] = {}
        animal_df = data[data["Animal"].isin(animal_list)]

        for condition in c1:

            all_animals_data[animal][condition] = {}

            df = update_df_with_epochs(
                animal_df[animal_df["Context"] == condition],
                pre_fmin,
                pre_fmax,
                epoch_tmin,
                epoch_tmax,
                "end",
            )
            df = df[df["epoch"].notnull()]

            # one baseline/epoch item for each session/row
            baseline = df["baseline"].tolist()
            baseline = np.array(
                [b[:-1] for b in baseline]
            )  # remove the last channel (ref)

            epochs = list(map(lambda e: set_event_id(e), df["epoch"].tolist()))
            fs = epochs[0].info["sfreq"]
            epochs = np.array(epochs)

            for ch_pair in combinations(range(len(brain_regions_all)), 2):

                pair_name = (
                    brain_regions_all[ch_pair[0]],
                    brain_regions_all[ch_pair[1]],
                )
                group = determine_group(pair_name)

                if group == "within":
                    continue

                all_animals_data[animal][condition][group] = {}

                connectivity_epochs = [
                    compute_coherence_base(
                        epoch, ch_pair, np.arange(pre_fmin, pre_fmax), fs
                    )
                    for epoch in epochs
                ]
                coherence_epochs = [
                    np.squeeze(x.get_data()) for x in connectivity_epochs
                ]
                connectivity_baseline = [
                    compute_coherence_base(
                        baseline, ch_pair, np.arange(pre_fmin, pre_fmax), sfreq
                    )
                    for bl in baseline
                ]
                coherence_baseline = [
                    np.squeeze(x.get_data()) for x in connectivity_baseline
                ]

                connectivity_freqs = [x.freqs for x in connectivity_epochs][0]

                mean_coherence_change, sem_coherence_change = average_coherence_change(
                    coherence_baseline, coherence_epochs
                )

                all_animals_data[animal][condition][group][
                    "mean"
                ] = mean_coherence_change
                all_animals_data[animal][condition][group]["sem"] = sem_coherence_change

                mean_coherence_change_1d = np.mean(mean_coherence_change, axis=1)
                sem_coherence_change_1d = np.mean(sem_coherence_change, axis=1)

    this_raw = data["raw"]
    this_processed = preprocess_raw(this_raw, pre_fmin, pre_fmax)
    this_processed.filter(15, 30, fir_design="firwin")
    this_processed.resample(1000)
    this_processed.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    this_processed.load_data()

    # average all animals in all_animals_data[animal][condition][group]['mean'] and ['sem']
    context_mean = np.mean(
        [
            all_animals_data[animal]["context"]["between"]["mean"]
            for animal in all_animals_data
        ],
        axis=0,
    )
    context_sem = np.mean(
        [
            all_animals_data[animal]["context"]["between"]["sem"]
            for animal in all_animals_data
        ],
        axis=0,
    )

    nocontext_mean = np.mean(
        [
            all_animals_data[animal]["nocontext"]["between"]["mean"]
            for animal in all_animals_data
        ],
        axis=0,
    )
    nocontext_sem = np.mean(
        [
            all_animals_data[animal]["nocontext"]["between"]["sem"]
            for animal in all_animals_data
        ],
        axis=0,
    )

    mean_per_freq_context = np.mean(context_mean, axis=1)
    sem_per_freq_context = np.mean(context_sem, axis=1)

    mean_per_freq_nocontext = np.mean(nocontext_mean, axis=1)
    sem_per_freq_nocontext = np.mean(nocontext_sem, axis=1)

    fig, ax = plt.subplots()
    ax.plot(connectivity_freqs, mean_per_freq_context, label="context", color="black")
    ax.fill_between(
        connectivity_freqs,
        mean_per_freq_context - sem_per_freq_context,
        mean_per_freq_context + sem_per_freq_context,
        alpha=0.2,
        color="black",
    )
    ax.plot(connectivity_freqs, mean_per_freq_nocontext, label="nocontext", color="red")
    ax.fill_between(
        connectivity_freqs,
        mean_per_freq_nocontext - sem_per_freq_nocontext,
        mean_per_freq_nocontext + sem_per_freq_nocontext,
        alpha=0.2,
        color="red",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence Change from Baseline (%)")

    ax.axhline(y=12, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=12, frameon=False)
    plt.show()
    from cpl_extract.viz import plots

    mydata = data[data["Animal"].isin(animal_list)].iloc[1]
    raw_o = mydata["raw"].copy()
    raw_o.pick_channels(["LFP1_vHp", "LFP3_AON"])
    # raw_o = preprocess_raw(raw_o, pre_fmin, pre_fmax)
    raw = raw_o.copy()
    raw.load_data()
    fs = raw.info["sfreq"]
    raw = raw.get_data()
    raw = raw[:-1, :]  # remove the last channel (ref)

    # get first epoch times and slice the array for the first trial
    first_event_times = [event[0][0] for event in df["end"].tolist()]
    first_event_times = [np.array([time]) for time in first_event_times]
    baseline_window = raw_o.crop(tmin=0, tmax=10)
    baseline_data = baseline_window.get_data()

    # get a trace 0.5s before and after the first event, just 1 row (channel)
    trace = raw[
        :, int(first_event_times[0] - 0.5 * fs) : int(first_event_times[0] + 0.5 * fs)
    ]
    trace = trace[0, :]

    raw = raw_o.copy()
    # Pick your channels of interest
    raw.pick_channels(["LFP1_vHp", "LFP3_AON"])
    raw.load_data()
    raw.filter(pre_fmin, pre_fmax, fir_design="firwin")
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    raw.load_data()

    # Define the beta frequency range
    beta_freq = (15, 30)

    # Filter the data for the beta frequency range
    raw_temp = mydata["raw"].copy()
    raw_temp.load_data()
    raw_beta = raw_temp.copy().filter(beta_freq[0], beta_freq[1], fir_design="firwin")

    # Get the data array from the raw and filtered objects
    data = raw_temp.get_data()
    data_beta = raw_beta.get_data()

    # Define the time range for the example trace (in seconds)
    start_time, end_time = 0, 40  # for instance, first 10 seconds

    # Define the smaller segment for beta activity visualization (in seconds)
    small_start, small_end = 0, 2  # for instance, a segment from 2 to 4 seconds
    # Times array
    times = raw.times

    # Find index range for the full trace and the smaller segment
    full_range = np.where((times >= start_time) & (times <= end_time))[0]

    afull_range = np.where((times >= 0) & (times <= 60))[0]
    small_range = np.where((times >= small_start) & (times <= small_end))[0]

    plt.close("all")
    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    # Plot the full trace
    ax[0].plot(
        times[full_range],
        data[0, full_range],
        label="Raw LFP",
        color="black",
        linewidth=2,
    )
    ax[0].plot(
        times[full_range],
        data_beta[0, full_range],
        label="Beta Filtered",
        color="red",
        linewidth=2,
    )
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")

    # Plot the smaller segment with the beta filtered data superimposed
    ax[1].plot(
        times[small_range],
        data[0, small_range],
        label="Raw LFP",
        color="black",
        linewidth=2,
    )
    ax[1].plot(
        times[small_range],
        data_beta[0, small_range],
        label="Beta Filtered",
        color="red",
        linewidth=2,
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Voltage (V)")

    # Adding legends
    ax[0].legend()

    plt.tight_layout()
    path = Path().home() / "data" / "plots"
    plt.savefig(
        f"{path}_beta_filtering_3.png",
        dpi=600,
        transparent=True,
    )
    plt.show()
    plt.close("all")

    # extract the relevant segment of data
    start_sample = int(start_time * raw.info["sfreq"])
    end_sample = int(end_time * raw.info["sfreq"])
    data_segment = raw.get_data(start=start_sample, stop=end_sample)
    data_beta_segment = raw_beta.get_data(start=start_sample, stop=end_sample)

    # compute the psd for the raw and beta-filtered data
    psd_raw, freqs = mne.time_frequency.psd_array_welch(
        data_segment,
        sfreq=raw.info["sfreq"],
        fmin=1,
        fmax=100,
        n_fft=int(raw.info["sfreq"]),
    )
    psd_beta, _ = mne.time_frequency.psd_array_welch(
        data_beta_segment,
        sfreq=raw_beta.info["sfreq"],
        fmin=1,
        fmax=100,
        n_fft=int(raw_beta.info["sfreq"]),
    )

    # Assuming psd_raw, psd_beta, and freqs are already computed and available
    mean_psd_raw = np.mean(psd_raw, axis=0)
    mean_psd_beta = np.mean(psd_beta, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot lines
    ax.plot(freqs, mean_psd_raw, label="Raw LFP", color="black", linewidth=3)
    ax.plot(freqs, mean_psd_beta, label="Beta Filtered", color="red", linewidth=3)

    # Fill between lines
    ax.fill_between(freqs, mean_psd_raw, mean_psd_beta, color="gray", alpha=0.3)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.legend()

    plt.show()

    raw1 = mydata["raw"].copy()
    fs = raw1.info["sfreq"]

    raw2 = mydata["raw"].copy()
    raw1.pick_channels(["LFP1_vHp", "LFP3_AON"])
    raw2.pick_channels(["LFP2_vHp", "LFP4_AON"])
    raw2 = preprocess_raw(raw1, pre_fmin, pre_fmax)
    raw2.filter(15, 30, fir_design="firwin")

    # Trimming raw1 to match the length of raw3
    if len(raw1.times) > len(raw2.times):
        raw1.crop(tmax=raw2.times[-1])

    # Extracting data
    data_raw1 = raw1.get_data()
    data_raw3 = raw2.get_data()

    # Calculating the difference
    difference = data_raw1 - data_raw3

    # Plot the difference
    times = raw1.times  # Using the time vector from raw1
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, difference.T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude Difference")
    ax.set_title("Difference between Original and Preprocessed Signals")
    plt.show()

    # avg_coherence_changes = {cond: np.mean(coherence_changes[], axis=0) for cond in coherence_changes}

    # sfreq = epochs[0].info['sfreq']
    # numt = epochs[0].times.size
    # params = optimize_psd_params(sfreq, numt)
    #
    # power_epochs = [compute_power(x, sfreq, *params) for x in epochs]
    # power_epochs_data = [x[0] for x in power_epochs]
    #
    # power_baseline = [compute_power(bl, sfreq, *params) for bl in baseline]
    # power_baseline_data = [bl[0] for bl in power_baseline]
    # power_freqs = power_baseline_data[0][0]
    #
    # pct_change_power = []
    #
    # for i, (baseline_power, epoch_power) in enumerate(zip(power_baseline_data, power_epochs_data)):
    #
    #     mean_bl_power = np.mean(baseline_power, axis=1)
    #     sem_bl_power = sem(baseline_power, axis=1)
    #
    #     percent_change_power = (epoch_power - mean_bl_power[:, np.newaxis]) / mean_bl_power[:, np.newaxis] * 100
    #     pct_change_power.append(percent_change_power)
    #
    #     # Calculate the mean and SEM for the percentage change
    #     mean_percent_change_power = np.mean(percent_change_power, axis=0)
    #     sem_percent_change_power = sem(percent_change_power, axis=0)

    # For statistical comparison, we perform a paired t-test between the baseline and each epoch's gamma power
    # t_stat, p_values = ttest_rel(baseline_power, epoch_power, axis=0)

    # ax.plot(power_freqs, mean_percent_change_gamma, label='Mean Percent Change')
    # ax.fill_between(power_freqs, mean_percent_change_gamma - sem_percent_change_gamma,
    #                 mean_percent_change_gamma + sem_percent_change_gamma, alpha=0.2)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Percentage Change in Gamma Power')
    # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(power_freqs, percent_change_power, label='mean')
    # ax.fill_between(power_freqs, percent_change_power, percent_change_power + sem_bl_power, alpha=0.2)
    # ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('Power Spectral Density ($V^2/Hz$)')
    # plt.show()
    # all_animals_data[animal][condition][group] = {}
    # for band, (fmin, fmax) in lfp_bands.items():
    #     idx = np.where((connectivity_freqs >= fmin) & (connectivity_freqs <= fmax))[0]  # x, 0 idx
    #
    #     mean_coh = np.mean(pct_change_power[idx, :], axis=0)  # mean across freqs
    #     sem_coh = np.std(pct_change_power[idx, :], axis=0) / np.sqrt(pct_change_power[idx, :].shape[0])
    #
    #     freqs_coh = connectivity_freqs[idx]
    #     all_animals_data[animal][condition][group][band] = {
    #         'mean': mean_coh,
    #         'sem': sem_coh,
    #         'freqs': freqs_coh,
    #         'baseline': df['baseline'].tolist()
    #     }
    #
    #     all_animals_data[animal][condition][group]['all'] = {
    #         'mean': coh,
    #         'mean_start': coh_start,
    #         'sem': sem_coh,
    #         'sem_start': sem_coh_start,
    #         'freqs': connectivity_freqs
    #     }

    #         # Plotting
    #     def plot():
    #         band_color_map = {'beta': 'r', 'gamma': 'b', 'theta': 'g'}
    #         plots_path = Path().home() / 'data' / 'plots' / 'coh2'
    #         plots_path.mkdir(parents=True, exist_ok=True)
    #
    #         for context in ["context", "nocontext"]:
    #
    #             plots_path_context = plots_path / context
    #             plots_path_context.mkdir(parents=True, exist_ok=True)
    #
    #             for group in ['between', 'within']:
    #                 plots_path_group = plots_path_context / group
    #                 plots_path_group.mkdir(parents=True, exist_ok=True)
    #
    #                 fig, (ax_time, ax_freq, ax_all) = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    #                 fig.suptitle(f'{animal} - All {context} trials - {group} brain regions - 1s pre dig', fontsize=16,
    #                              fontweight='bold')
    #
    #                 for domain, ax in [('time_domain', ax_time), ('freq_domain', ax_freq)]:
    #
    #                     ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    #                     ax.grid(False)
    #
    #                     if domain == 'time_domain':
    #                         ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    #                         ax.set_ylabel('Coherence', fontsize=14, fontweight='bold')
    #                         ax.set_ylim(0, 1)
    #
    #                     elif domain == 'freq_domain':
    #                         ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    #                         ax.set_ylabel('Coherence', fontsize=14, fontweight='bold')
    #                         ax.set_ylim(0, 1)
    #
    #                     for band_name, coh_data in all_animals_data[animal][context][group][domain].items():
    #                         y_mean = coh_data['mean']
    #                         y_sem = coh_data['sem']
    #
    #                         if domain == 'time_domain':
    #                             x = connectivity_epochs.times
    #                         else:
    #                             x = coh_data['freqs']
    #
    #                         color = band_color_map[band_name]
    #
    #                         # Plot the mean and SEM shading
    #                         ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, alpha=0.2, color=color)
    #                         ax.plot(x, y_mean, linewidth=2, label=f'{band_name}', color=color, )
    #
    #                 y_mean_all = all_animals_data[animal][context][group]['all']['mean']
    #                 x_all = connectivity_epochs.times
    #
    #                 spectro = ax_all.imshow(y_mean_all, cmap='jet', aspect='auto',
    #                                         extent=[x_all[0], x_all[-1], 0, 1])
    #
    #                 ax_all.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    #                 ax_all.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    #
    #                 ax_all.set_title(f"Spatio-Temporal Connectivity", fontsize=14, fontweight='bold')
    #                 ax_all.grid(False)
    #
    #                 plt.colorbar(spectro, ax=ax_all, label='Coherence')
    #
    #                 ax_time.set_ylabel('Coherence', fontsize=14, fontweight='bold')
    #                 ax_time.set_ylim(0, 1)
    #
    #                 # Add legends to the plots
    #                 ax_time.legend(loc='best')
    #                 ax_freq.legend(loc='best')
    #
    #                 # Display the plot
    #                 # plt.show()
    #                 plt.savefig(plots_path_group / f'{animal}_{context}_{group}_digwindow.png', dpi=300,
    #                             bbox_inches='tight',
    #                             pad_inches=0.1, )
    # #
    # # # plt.figure(figsize=(11, 6))
    # # # plt.title(f'Average Beta Power Spectra: {animal}')
    # # #
    # # # plt.fill_between(freqs[beta_indices],
    # # #                  beta_mean_psd_aon_context - beta_sem_psd_aon_context,
    # # #                  beta_mean_psd_aon_context + beta_sem_psd_aon_context,
    # # #                  alpha=0.2)
    # # # plt.fill_between(freqs[beta_indices],
    # # #                  beta_mean_psd_aon_nocontext - beta_sem_psd_aon_nocontext,
    # # #                  beta_mean_psd_aon_nocontext + beta_sem_psd_aon_nocontext,
    # # #                  alpha=0.2)
    # # # plt.plot(freqs[beta_indices], beta_mean_psd_aon_context, linewidth=2, label='Beta - Context')
    # # # plt.plot(freqs[beta_indices], beta_mean_psd_aon_nocontext, linewidth=2, label='Beta - No Context',
    # # #          linestyle='--')
    # # # plt.fill_between(freqs[beta_indices],
    # # #                  beta_mean_psd_vhp_context - beta_sem_psd_vhp_context,
    # # #                  beta_mean_psd_vhp_context + beta_sem_psd_vhp_context,
    # # #                  alpha=0.2)
    # # # plt.fill_between(freqs[beta_indices],
    # # #                  beta_mean_psd_vhp_nocontext - beta_sem_psd_vhp_nocontext,
    # # #                  beta_mean_psd_vhp_nocontext + beta_sem_psd_vhp_nocontext,
    # # #                  alpha=0.2)
    # # # plt.plot(freqs[beta_indices], beta_mean_psd_vhp_context, linewidth=2, label='Beta - Context')
    # # # plt.plot(freqs[beta_indices], beta_mean_psd_vhp_nocontext, linewidth=2, label='Beta - No Context',)
    # # # plt.xlabel('Frequency (Hz)')
    # # # plt.ylabel('Power Spectral Density ($V^2/Hz$)')
    # # # plt.legend()
    # # # plt.show()
    # # #
    # # # # Plotting
    # # # plt.figure(figsize=(10, 6))
    # # #
    # # # # Plot AON Context vs No Context
    # # # plt.fill_between(freqs, aon_mean_psd_context - aon_sem_psd_context, aon_mean_psd_context + aon_sem_psd_context,
    # # #                  alpha=0.2)
    # # # plt.fill_between(freqs, aon_mean_psd_nocontext - aon_sem_psd_nocontext,
    # # #                  aon_mean_psd_nocontext + aon_sem_psd_nocontext, alpha=0.2,)
    # # # plt.plot(freqs, aon_mean_psd_context, linewidth=2, label='AON - Context', )
    # # # plt.plot(freqs, aon_mean_psd_nocontext, linewidth=2, label='AON - No Context', linestyle='--')
    # # #
    # # # # Plot vHp Context vs No Context
    # # # plt.fill_between(freqs, vhp_mean_psd_context - vhp_sem_psd_context, vhp_mean_psd_context + vhp_sem_psd_context,
    # # #                  alpha=0.2,)
    # # # plt.fill_between(freqs, vhp_mean_psd_nocontext - vhp_sem_psd_nocontext,
    # # #                  vhp_mean_psd_nocontext + vhp_sem_psd_nocontext, alpha=0.2,)
    # # # plt.plot(freqs, vhp_mean_psd_context, label='vHp - Context', linewidth=2)
    # # # plt.plot(freqs, vhp_mean_psd_nocontext, label='vHp - No Context', linewidth=2, linestyle='--')
    # # #
    # # # plt.xlabel('Frequency (Hz)')
    # # # plt.ylabel('Power Spectral Density ($V^2/Hz$)')
    # # # plt.legend()
    # # #
    # # # plt.title(f'Overlaid Average Power Spectra Across Trials: {animal}')
    # # # plt.show()
    # #
    # # # indices = (np.array([0]), np.array([1]))
    # # # freqs = np.arange(fmin, fmax, 11)
    # # #
    # # # con_con = spectral_connectivity_epochs(
    # # #     con_concat,
    # # #     indices=indices,
    # # #     method="coh",
    # # #     mode="cwt_morlet",
    # # #     cwt_freqs=freqs,
    # # #     cwt_n_cycles=freqs / 2,
    # # #     sfreq=con_concat.info["sfreq"],
    # # #     n_jobs=1,
    # # # )
    # # # noncon_con = spectral_connectivity_epochs(
    # # #     nocon_concat,
    # # #     indices=indices,
    # # #     method="coh",
    # # #     mode="cwt_morlet",
    # # #     cwt_freqs=freqs,
    # # #     cwt_n_cycles=freqs / 2,
    # # #     sfreq=con_concat.info["sfreq"],
    # # #     n_jobs=1,
    # # # )
    # # # con_coh = con_con.get_data()
    # # # z_scores = np.abs(stats.zscore(con_coh, axis=2))
    # # # outliers = (z_scores > 2).all(axis=2)
    # # # con_coh = con_coh[~outliers]
    # # #
    # # # noncon_coh = noncon_con.get_data()
    # # # z_scores = np.abs(stats.zscore(noncon_coh, axis=2))
    # # # outliers = (z_scores > 2).all(axis=2)
    # # # noncon_coh = noncon_coh[~outliers]
    # # #
    # # # con_con_avg = np.mean(con_coh, axis=1)
    # # # noncon_con_avg = np.mean(noncon_coh, axis=1)
    # # #
    # # # fig, ax = plt.subplots(figsize=(12, 8))
    # # # ax.plot(freqs, con_con_avg,
    # # #         label="Context")
    # # # ax.plot(freqs, noncon_con_avg, label="No Context")
    # # # ax.set_xlabel("Frequency (Hz)", fontsize=14)
    # # # ax.set_ylabel("Coherence", fontsize=14)
    # # # ax.set_title("Context vs No Context", fontsize=16, fontweight="bold")
    # # # ax.legend()
    # # # plt.tight_layout()
    # # # plt.show()
    # # #
    # # # x=4
    # #
    # # # plots_path = Path().home() / "data" / "plots" / "aon"
    # # # plots_path.mkdir(parents=True, exist_ok=True)
    # # # data = get_master_df()
    # # # data.reset_index(drop=True, inplace=True)
    # # #
    # # # num_sessions = data.shape[0]
    # # # # get the timestamps when for each key in the event_id dict
    # # # for i in range(num_sessions):
    # # #     animal_path = plots_path / f"{data.iloc[i]['Animal']}"
    # # #     session_path = (
    # # #         animal_path
    # # #         / f"{data.iloc[i]['Date']}_{data.iloc[i]['Day']}_{data.iloc[i]['Stimset']}_{data.iloc[i]['Context']}"
    # # #     )
    # # #     session_path.mkdir(parents=True, exist_ok=True)
    # # #
    # # #     row = data.iloc[i, :]
    # # #     id_dict = row["event_id"]
    # # #     raw: mne.io.RawArray = row["raw"]
    # # #     start = row["start"]
    # # #     end = row["end"]
    # # #     start_time = start[0, 0] / 1000
    # # #
    # # #     raw.load_data()
    # # #     raw_copy: mne.io.RawArray = raw.copy()
    # # #
    # # #     raw_copy.apply_function(zscore)
    # # #     raw_copy.apply_function(robust_scale)
    # # #
    # # #     raw_copy.filter(0.3, 100, fir_design="firwin")
    # # #     raw_copy.notch_filter(freqs=np.arange(60, 121, 60))
    # # #     raw_copy.set_eeg_reference(ref_channels=["Ref"])
    # # #     raw_copy.drop_channels(["Ref"])
    # # #
    # # #     iter_freqs = [
    # # #         ("Beta", 13, 25),
    # # #         ("Gamma", 30, 45),
    # # #     ]
    # # #     freqs = [(13, 25), (30, 45)]
    # # #     coh_data = {}
    # # #
    # # #     tmin, tmax = -1, 0  # time window (seconds)
    # # #     chans = ["LFP1_vHp", "LFP4_AON"]
    # # #     for band, fmin, fmax in iter_freqs:
    # # #         band_path = session_path / band
    # # #         band_path.mkdir(parents=True, exist_ok=True)
    # # #
    # # #         raw_copy.pick_channels(chans)
    # # #         raw_copy.filter(
    # # #             fmin,
    # # #             fmax,
    # # #             n_jobs=None,
    # # #             l_trans_bandwidth=1,
    # # #             h_trans_bandwidth=1,
    # # #         )
    # # #         epoch = mne.Epochs(
    # # #             raw_copy,
    # # #             row["end"],
    # # #             row["event_id"],
    # # #             tmin,
    # # #             tmax,
    # # #             baseline=None,
    # # #             event_repeated="drop",
    # # #             preload=True,
    # # #         )
    # # #
    # # #         freqs_arange = np.arange(fmin, fmax)
    # # #         con = spectral_connectivity_epochs(
    # # #             epoch,
    # # #             indices=(np.array([0]), np.array([1])),
    # # #             method="coh",
    # # #             mode="cwt_morlet",
    # # #             cwt_freqs=freqs_arange,
    # # #             cwt_n_cycles=freqs_arange / 2,
    # # #             sfreq=epoch.info["sfreq"],
    # # #             n_jobs=1,
    # # #         )
    # # #         times = epoch.times
    # # #         coh_data = np.squeeze(con.get_data())
    # # #
    # # #         title = f"{band} Coherence {chans[0]} vs {chans[1]}"
    # # #         filename = f"{band}_{chans[0]}_{chans[1]}.png"
    # # #         plots.plot_2D_coherence(
    # # #             coh_data,
    # # #             times,
    # # #             np.arange(fmin, fmax),
    # # #             title,
    # # #             filename,
    # # #             band_path,
    # # #         )
    # # #         plot_epoch_data(epoch)
