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
from scipy.stats import zscore, sem
from sklearn.manifold import spectral_embedding  # noqa

from cpl_pipeline.analysis.stats import extract_file_info, extract_file_info_on
from cpl_pipeline.logs import logger
from cpl_pipeline.spk_io.utils import read_npz_as_dict
from cpl_pipeline.utils import extract_common_key
from cpl_pipeline.viz import helpers
from cpl_pipeline.viz import plots

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
    data_path = Path().home() / "data" / ".cache" / "serotonin"
    # prevent .DS_Store from being read
    animals = [animal for animal in data_path.glob("*") if animal.is_dir()]
    master = pd.DataFrame()
    for animal_dir in animals:
        cache_animal_path = data_path / animal_dir.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        lfp_raw = sorted(
            list(animal_dir.glob("*_lfp_raw*.fif")), key=extract_common_key
        )
        unit_raw = sorted(
            list(animal_dir.glob("*_unit_raw*.fif")), key=extract_common_key
        )
        event_files = sorted(
            list(animal_dir.glob("*_eve*.fif")), key=extract_common_key
        )
        event_id = sorted(list(animal_dir.glob("*_id_ev*.npz")), key=extract_common_key)
        sniff_signal = sorted(
            list(animal_dir.glob("*_sniff_signal*.npy")), key=extract_common_key
        )
        sniff_times = sorted(
            list(animal_dir.glob("*_sniff_times*.npy")), key=extract_common_key
        )

        metadata = [extract_file_info_on(raw_file.name) for raw_file in lfp_raw]
        lfp_data = [mne.io.read_raw_fif(raw_file) for raw_file in lfp_raw]
        unit_data = [mne.io.read_raw_fif(raw_file) for raw_file in unit_raw]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        event_id_dicts = [read_npz_as_dict(id_file) for id_file in event_id]
        sniff_signal = [np.load(str(sniff_file)) for sniff_file in sniff_signal]
        sniff_times = [np.load(str(sniff_file)) for sniff_file in sniff_times]

        first_event_times = [event[0][0] for event in event_data]
        sfreq = lfp_data[0].info["sfreq"]

        baseline_segment = [get_lowest_variance_window(raw, sfreq) for raw in lfp_data]
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

        # TODO: this should be user provided based on needed information
        all_data_dict = {  # each v is a list for each file, sorted to be the smae order
            "raw": lfp_data,
            "raw_unit": unit_raw,
            "start": ev_start_holder,
            "sniff_sig": sniff_signal,
            "sniff_times": sniff_times,
            "event_id": ev_id_dict,
            "baseline": baseline_segment,
        }
        # pop first item in each list with 4 items

        metadata_df = pd.concat(metadata).reset_index(drop=True)
        all_data_df = pd.DataFrame(all_data_dict)
        all_data_df = pd.concat([metadata_df, all_data_df], axis=1)
        master = pd.concat([master, all_data_df], axis=0)

    return master


def preprocess_raw(raw_signal, fmin=1, fmax=100):
    raw: mne.io.RawArray = raw_signal
    raw.load_data()
    raw.filter(fmin, fmax, fir_design="firwin")
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
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
    assert fmin < fmax and tmin < tmax, "Ensure fmin < fmax and tmin < tmax"

    raw = preprocess_raw(raw_arr, fmin, fmax)
    means = np.mean(baseline, axis=1)
    stds = np.std(baseline, axis=1)

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


def compute_coherence_base(epoch_obj, idxs, freqs):
    epoch_arr = epoch_obj.get_data()
    epoch_connectivity = spectral_connectivity_epochs(
        epoch_arr,
        indices=(np.array([idxs[0]]), np.array([idxs[1]])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=freqs / 2,
        sfreq=epoch_obj.info["sfreq"],
        n_jobs=1,
    )
    return epoch_connectivity


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
    data = data.dropna(subset=["raw"])
    day = data.iloc[0]

    epoch_tmin, epoch_tmax = -1, 0
    epoch_start_tmin, epoch_start_tmax = 0, 1

    pre_fmin, pre_fmax = 1, 100

    # TODO: user provided information, or programmatically derived
    brain_regions_all = ["LFP1_OB", "LFP2_OB", "LFP1_PC"]
    brain_regions_within = [("LFP1_OB", "LFP2_OB")]
    brain_regions_between = [("LFP1_OB", "LFP1_PC"), ("LFP2_OB", "LFP1_PC")]

    all_animals = data["Animal"].unique()

    # TODO: user provided information, or programmatically derived
    all_event_keys = {"O_o": 1}
    lfp_bands = {"theta": (4, 12)}

    chan_indices = np.array(list(combinations(range(4), 2)))

    sniff_signal = day["sniff_sig"]
    sniff_times = day["sniff_times"]
    print(sniff_signal.shape)
    print(sniff_times.shape)

    raw = day["raw"]
    unit = day["raw_unit"]
    event_id = day["event_id"]
    baseline = day["baseline"]

    # Detect peaks
    peaks, _ = scipy.signal.find_peaks(sniff_signal)

    # Convert peak indices to time (if you know the sampling rate)
    peak_times = peaks  # If the sampling rate is unknown, you can only get the indices

    # Plot the results (optional)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(sniff_signal[peaks[:100000]])
    plt.plot(peaks[:1000], sniff_signal[peaks[:1000]], "x", color="black")
    plt.title("Sniff Detection")
    plt.xlabel("Samples")
    plt.ylabel("Pressure")
    plt.show()

    x = 5
