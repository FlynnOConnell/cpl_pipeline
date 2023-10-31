from __future__ import annotations
import time

from fooof import FOOOF
from scipy.ndimage import median_filter
from scipy import signal
from scipy.stats import ttest_ind, stats

import functools
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.cm
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, gridspec
from mne.time_frequency import psd_array_welch
from mne_connectivity import spectral_connectivity_epochs
from scipy.interpolate import interp1d
from scipy.signal import coherence
from scipy.stats import zscore, ttest_rel
from sklearn.manifold import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel

from spk2extract.logs import logger
from spk2extract.viz import helpers

sns.set_style("darkgrid")

# # get the timestamps when for each key in the event_id dict
# for key in id_dict.keys():
#     if key in ["b_1", "b_0", "w_1", "w_0", "x_1", "x_0"]:
#         continue
#     else:
#         val = id_dict[key]
#         starts = start[val == start[:, 2], 0]
#         ends = end[val == end[:, 2], 0]
#         starts = starts / 1000
#         ends = ends / 1000
#         zipped = zip(starts, ends)
# print("------------------")
# print(f"animal {row['Animal']} - {row['Date']}")
# print(f"{key} : {list(zipped)}")

logger.setLevel("INFO")

def pad_arrays_to_same_length(arr_list, max_diff=100):
    """
    Pads numpy arrays to the same length.

    Parameters:
    - arr_list (list of np.array): The list of arrays to pad
    - max_diff (int): Maximum allowed difference in lengths

    Returns:
    - list of np.array: List of padded arrays
    """
    lengths = [len(arr) for arr in arr_list]
    max_length = max(lengths)
    min_length = min(lengths)

    if max_length - min_length > max_diff:
        raise ValueError("Arrays differ by more than the allowed maximum difference")

    padded_list = []
    for arr in arr_list:
        pad_length = max_length - len(arr)
        padded_arr = np.pad(arr, (0, pad_length), "constant", constant_values=0)
        padded_list.append(padded_arr)

    return padded_list

def extract_file_info(filename):
    parts = filename.split("_")
    data = {
        "Date": None,
        "Animal": None,
        "Type": "None",
        "Context": "None",
        "Day": "1",
        "Stimset": "1",
    }

    has_date = False
    has_animal = False

    for part in parts:
        if not has_date and re.match(r"\d{6,8}", part):
            data["Date"] = pd.to_datetime(part, format="%Y%m%d").strftime("%Y-%m-%d")
            has_date = True
        elif not has_animal and re.match(r"[a-zA-Z]+\d+$", part):
            data["Animal"] = part
            has_animal = True
        elif re.match(r"BW", part):
            data["Type"] = "BW"
        elif re.match(r"(nocontext|context)", part):
            data["Context"] = part
        elif re.match(r"(day\d+|d\d+)", part, re.IGNORECASE):
            data["Day"] = re.findall(r"\d+", part)[0]
        elif re.match(r"os\d+", part):
            data["Stimset"] = re.findall(r"\d+", part)[0]

    if not data["Date"] or not data["Animal"]:
        data = {k: "error" for k in data.keys()}

    return pd.DataFrame([data])

def extract_common_key(filepath):
    parts = filepath.stem.split("_")
    return "_".join(parts[:-1])

def read_npz_as_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as npz_data:
        return {k: npz_data[k] for k in npz_data.keys()}

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

def get_baseline(resp_signal, resp_times, wave_signal, first):
    wave_signal = wave_signal.get_data()
    first_event_time = first
    baseline_size = 10
    respiratory_signal = np.array(resp_signal[:int(first_event_time)], dtype=float)
    respiratory_times = np.array(resp_times[:int(first_event_time)], dtype=float)
    wave_signal = np.array(wave_signal[:, :int(first_event_time * 2000)], dtype=float)

    # Step 1: Identify slow breathing segment using rolling variance
    rolling_variance = pd.Series(respiratory_signal).rolling(window=baseline_size).var()
    rolling_variance = rolling_variance.dropna()
    min_var_index = rolling_variance.idxmin()
    slowest_breathing_time = respiratory_times[min_var_index]

    # Convert this time to an index in spikes_arr
    slowest_breathing_index_spikes_arr = int(
        slowest_breathing_time * 2000)  # Conversion factor due to different sampling rates

    # Step 2: Extract corresponding segment from spikes_arr
    segment_data = wave_signal[:,
                   slowest_breathing_index_spikes_arr:slowest_breathing_index_spikes_arr + baseline_size * 2000]

    return segment_data

def get_master_df():
    cache_path = Path().home() / "data" / ".cache"
    animals = list(cache_path.iterdir())

    master = pd.DataFrame()
    for animal_path in animals:  # each animal has a folder
        cache_animal_path = cache_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(list(animal_path.glob("*_raw*.fif")), key=extract_common_key)
        event_files = sorted(list(animal_path.glob("*_eve*.fif")), key=extract_common_key)
        event_id = sorted(list(animal_path.glob("*_id_*.npz")), key=extract_common_key)
        norm_signal = sorted(animal_path.glob("*_respiratory_signal*.npy"), key=extract_common_key)
        norm_times = sorted(animal_path.glob("*_respiratory_times*.npy"), key=extract_common_key)
        assert len(raw_files) == len(event_files) == len(event_id)

        metadata = [extract_file_info(raw_file.name) for raw_file in raw_files]
        raw_data = [mne.io.read_raw_fif(raw_file) for raw_file in raw_files]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        event_id_dicts = [read_npz_as_dict(id_file) for id_file in event_id]

        norm_signal = [np.load(str(signal_file)) for signal_file in norm_signal]
        norm_times = [np.load(str(time_file)) for time_file in norm_times]

        first_event_times = [event[0][0] for event in event_data]
        baseline_segment = [get_baseline(resp_signal, resp_times, raw_arr, first_event) for
                            resp_signal, resp_times, raw_arr, first_event in
                            zip(norm_signal, norm_times, raw_data, first_event_times)]
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
            "events": event_data,
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

def rolling_coherence(x, y, window, fs=1.0):
    coh_vals = []
    for i in range(0, len(x) - window, window // 2):  # 50% overlap
        f, Cxy = coherence(
            x[i: i + window], y[i: i + window], fs=fs, nperseg=window // 2
        )
        coh_vals.append(
            np.mean(Cxy)
        )  # Average coherence across frequencies, adjust as needed
    return np.array(coh_vals)

def plot_all_epoch(epoch, title, savepath=None):
    for epoch_idx, (epoch_signal_ch1, epoch_signal_ch2) in enumerate(epoch):
        window_size = 100
        coh_values = rolling_coherence(epoch_signal_ch1, epoch_signal_ch2, window_size)
        epoch_times = epoch.times
        coh_times = epoch_times[: len(coh_values)]

        interp_func = interp1d(coh_times, coh_values, kind="cubic")
        new_coh_times = np.linspace(coh_times[0], coh_times[-1], 500)
        new_coh_values = interp_func(new_coh_times)

        # Create grid and subplots
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 0.25, 3])
        gs.update(hspace=0.0)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        # General settings
        ax1.set_title(f"{title}, trial {epoch_idx}")
        for ax in [ax1, ax2]:
            ax.grid(False)

        # Plotting data
        colormap = matplotlib.colormaps["jet"]
        norm = plt.Normalize(0, 1.5)

        # Coherence as fill_between
        num_segments = len(new_coh_times) - 1
        for seg in range(num_segments):
            color_value = norm(new_coh_values[seg])
            ax1.fill_between(
                new_coh_times[seg: seg + 2],
                0,
                new_coh_values[seg: seg + 2],
                color=colormap(color_value),
            )

        # Coherence as heatmap
        ax2.imshow(
            [new_coh_values],
            aspect="auto",
            cmap=colormap,
            norm=norm,
            extent=[new_coh_times[0], new_coh_times[-1], 0, 1],
        )

        # Signal plot
        ax3.plot(
            epoch.times,
            epoch_signal_ch1 * 1000,
            linestyle="-",
            linewidth=1,
            color="black",
        )
        ax3.plot(
            epoch.times,
            epoch_signal_ch2 * 1000,
            linestyle="--",
            linewidth=1,
            color="r",
        )

        xlim = [new_coh_times[0], new_coh_times[-1]]
        ax1.set_xlim(xlim)
        ax1.set_ylim([0, 1.5])
        ax2.set_xlim(xlim)
        ax3.set_xlim([-1, 0])

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])

        xticks = np.linspace(-1, 0, 11)
        xticklabels = np.linspace(-1000, 0, 11).astype(int).tolist()
        xticklabels[-1] = "DIG"

        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xticklabels)

        ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        if savepath:
            plt.savefig(f"{savepath}/{epoch_idx}.png")
            logger.info(f"Saved {savepath}/{epoch_idx}.png")
            plt.close("all")
        else:
            plt.show()

def preprocess_raw(raw_signal, fmin=1, fmax=100):
    raw: mne.io.RawArray = raw_signal.copy()
    raw.load_data()
    raw.filter(fmin, fmax, fir_design="firwin")
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    raw.set_eeg_reference(ref_channels=["Ref"])
    raw.drop_channels(["Ref"])
    return raw

def get_filtered_epoch(raw_obj, events, event_id, fmin, fmax, tmin, tmax, baseline=None):
    raw = raw_obj.filter(fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1)
    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=baseline,
        event_repeated="drop",
        preload=True,
    )

def update_df_with_epochs(df, fmin, fmax, tmin, tmax):
    epochs_holder, con_holder = [], []
    for idx, row in df.iterrows():
        epoch, con_data = process_epoch(row['raw'], row['events'], row['event_id'], row['baseline'], fmin, fmax, tmin,
                                        tmax)
        epochs_holder.append(epoch)

        con_holder.append(con_data)
    df['epoch'] = epochs_holder
    df['con_arr'] = con_holder
    return df


def process_epoch(raw_arr, events, event_id, baseline, fmin, fmax, tmin, tmax):
    assert fmin < fmax and tmin < tmax, "Ensure fmin < fmax and tmin < tmax"
    raw = preprocess_raw(raw_arr, fmin, fmax)
    means = np.mean(baseline, axis=1)
    stds = np.std(baseline, axis=1)
    means = means[:-1]
    stds = stds[:-1]

    # Continue processing
    valid_events, valid_event_id = _validate_events(events, event_id)

    epochs = get_filtered_epoch(raw, valid_events, valid_event_id, fmin, fmax, tmin, tmax)
    if epochs is None:
        raise ValueError("Epoch is None")

    epochs_data = epochs.get_data()
    for ch in epochs.ch_names:
        ch_idx = epochs.ch_names.index(ch)
        epochs._data[:, ch_idx, :] = (epochs._data[:, ch_idx, :] - means[ch_idx]) / stds[ch_idx]

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


if __name__ == "__main__":
    data = main(use_parallel=False)

    tmin, tmax = -1, 0
    fmin, fmax = 1, 100

    fm = FOOOF()
    for animal in data["Animal"].unique():
        animal_df = data[data["Animal"] == animal]
        context_df = animal_df[animal_df["Context"] == "context"]
        no_context_df = animal_df[animal_df["Context"] == "nocontext"]

        updated_context_df = update_df_with_epochs(context_df, fmin, fmax, tmin, tmax)
        updated_no_context_df = update_df_with_epochs(no_context_df, fmin, fmax, tmin, tmax)

        epochs_context = updated_context_df['epoch'].tolist()
        epochs_nocontext = updated_no_context_df['epoch'].tolist()

        key = {"b_1": 1, "b_0": 2, "w_1": 3, "w_0": 4, "x_1": 5, "x_0": 6}

        for large_epoch in epochs_context:
            large_epoch.event_id = key
        for large_epoch in epochs_nocontext:
            large_epoch.event_id = key

        picks = ["LFP1_vHp", "LFP3_AON"]
        con_concat = mne.concatenate_epochs(epochs_context)
        nocon_concat: mne.Epochs = mne.concatenate_epochs(epochs_nocontext)
        con_concat.picks = picks
        nocon_concat.picks = picks

        # PSD calculation settings
        n_fft, n_per_seg, n_overlap = optimize_psd_params(con_concat.info["sfreq"], con_concat.get_data().shape[2])

        # Calculate and normalize the PSD
        aon = ['LFP3_AON']
        vhp = ['LFP1_vHp']
        title = f'Average Power Spectra: {animal}'
        subtitle = ''

        aon_data_context = con_concat.get_data(picks=aon)
        aon_data_nocontext = nocon_concat.get_data(picks=aon)

        vhp_data_context = con_concat.get_data(picks=vhp)
        vhp_data_nocontext = nocon_concat.get_data(picks=vhp)


        aon_psd_context, freqs = psd_array_welch(aon_data_context, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                                                 n_overlap=n_overlap, sfreq=con_concat.info["sfreq"])
        aon_psd_nocontext, _ = psd_array_welch(aon_data_nocontext, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                                               n_overlap=n_overlap, sfreq=nocon_concat.info["sfreq"])
        vhp_psd_context, _ = psd_array_welch(vhp_data_context, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                                             n_overlap=n_overlap, sfreq=con_concat.info["sfreq"])
        vhp_psd_nocontext, _ = psd_array_welch(vhp_data_nocontext, fmin=fmin, fmax=fmax, n_fft=n_fft, n_per_seg=n_per_seg,
                                               n_overlap=n_overlap, sfreq=nocon_concat.info["sfreq"])

        # Compute the mean and SEM
        aon_mean_psd_context = np.mean(aon_psd_context, axis=(0, 1))
        aon_mean_psd_nocontext = np.mean(aon_psd_nocontext, axis=(0, 1))
        aon_sem_psd_context = np.std(aon_psd_context, axis=(0, 1)) / np.sqrt(aon_psd_context.shape[0])
        aon_sem_psd_nocontext = np.std(aon_psd_nocontext, axis=(0, 1)) / np.sqrt(aon_psd_nocontext.shape[0])

        vhp_mean_psd_context = np.mean(vhp_psd_context, axis=(0, 1))
        vhp_mean_psd_nocontext = np.mean(vhp_psd_nocontext, axis=(0, 1))
        vhp_sem_psd_context = np.std(vhp_psd_context, axis=(0, 1)) / np.sqrt(vhp_psd_context.shape[0])
        vhp_sem_psd_nocontext = np.std(vhp_psd_nocontext, axis=(0, 1)) / np.sqrt(vhp_psd_nocontext.shape[0])

        # # Assuming baseline_aon and baseline_vhp contain the baseline PSD for AON and vHp
        # aon_rel_power_change_context = ((aon_mean_psd_context - baseline_aon) / baseline_aon) * 100
        # aon_rel_power_change_nocontext = ((aon_mean_psd_nocontext - baseline_aon) / baseline_aon) * 100
        #
        # vhp_rel_power_change_context = ((vhp_mean_psd_context - baseline_vhp) / baseline_vhp) * 100
        # vhp_rel_power_change_nocontext = ((vhp_mean_psd_nocontext - baseline_vhp) / baseline_vhp) * 100

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot AON Context vs No Context
        plt.fill_between(freqs, aon_mean_psd_context - aon_sem_psd_context, aon_mean_psd_context + aon_sem_psd_context,
                         alpha=0.2)
        plt.fill_between(freqs, aon_mean_psd_nocontext - aon_sem_psd_nocontext,
                         aon_mean_psd_nocontext + aon_sem_psd_nocontext, alpha=0.2,)
        plt.plot(freqs, aon_mean_psd_context, linewidth=2, label='AON - Context', )
        plt.plot(freqs, aon_mean_psd_nocontext, linewidth=2, label='AON - No Context', linestyle='--')

        # Plot vHp Context vs No Context
        plt.fill_between(freqs, vhp_mean_psd_context - vhp_sem_psd_context, vhp_mean_psd_context + vhp_sem_psd_context,
                         alpha=0.2,)
        plt.fill_between(freqs, vhp_mean_psd_nocontext - vhp_sem_psd_nocontext,
                         vhp_mean_psd_nocontext + vhp_sem_psd_nocontext, alpha=0.2,)
        plt.plot(freqs, vhp_mean_psd_context, label='vHp - Context', linewidth=2)
        plt.plot(freqs, vhp_mean_psd_nocontext, label='vHp - No Context', linewidth=2, linestyle='--')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density ($\mu V^2/Hz$)')
        plt.legend()

        plt.legend()
        plt.title(f'Overlaid Average Power Spectra Across Trials: {animal}' + '\n' + subtitle)
        # subtitle underneath title

        plt.tight_layout()
        plt.show()

        for baseline in data['baseline']:
            pass

    #
    # indices = (np.array([0]), np.array([1]))
    # freqs = np.arange(fmin, fmax, 11)
    #
    # con_con = spectral_connectivity_epochs(
    #     con_concat,
    #     indices=indices,
    #     method="coh",
    #     mode="cwt_morlet",
    #     cwt_freqs=freqs,
    #     cwt_n_cycles=freqs / 2,
    #     sfreq=con_concat.info["sfreq"],
    #     n_jobs=1,
    # )
    # noncon_con = spectral_connectivity_epochs(
    #     nocon_concat,
    #     indices=indices,
    #     method="coh",
    #     mode="cwt_morlet",
    #     cwt_freqs=freqs,
    #     cwt_n_cycles=freqs / 2,
    #     sfreq=con_concat.info["sfreq"],
    #     n_jobs=1,
    # )
    # con_coh = con_con.get_data()
    # z_scores = np.abs(stats.zscore(con_coh, axis=2))
    # outliers = (z_scores > 2).all(axis=2)
    # con_coh = con_coh[~outliers]
    #
    # noncon_coh = noncon_con.get_data()
    # z_scores = np.abs(stats.zscore(noncon_coh, axis=2))
    # outliers = (z_scores > 2).all(axis=2)
    # noncon_coh = noncon_coh[~outliers]
    #
    # con_con_avg = np.mean(con_coh, axis=1)
    # noncon_con_avg = np.mean(noncon_coh, axis=1)
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(freqs, con_con_avg,
    #         label="Context")
    # ax.plot(freqs, noncon_con_avg, label="No Context")
    # ax.set_xlabel("Frequency (Hz)", fontsize=14)
    # ax.set_ylabel("Coherence", fontsize=14)
    # ax.set_title("Context vs No Context", fontsize=16, fontweight="bold")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # x=4

    # plots_path = Path().home() / "data" / "plots" / "aon"
    # plots_path.mkdir(parents=True, exist_ok=True)
    # data = get_master_df()
    # data.reset_index(drop=True, inplace=True)
    #
    # num_sessions = data.shape[0]
    # # get the timestamps when for each key in the event_id dict
    # for i in range(num_sessions):
    #     animal_path = plots_path / f"{data.iloc[i]['Animal']}"
    #     session_path = (
    #         animal_path
    #         / f"{data.iloc[i]['Date']}_{data.iloc[i]['Day']}_{data.iloc[i]['Stimset']}_{data.iloc[i]['Context']}"
    #     )
    #     session_path.mkdir(parents=True, exist_ok=True)
    #
    #     row = data.iloc[i, :]
    #     id_dict = row["event_id"]
    #     raw: mne.io.RawArray = row["raw"]
    #     start = row["start"]
    #     end = row["end"]
    #     start_time = start[0, 0] / 1000
    #
    #     raw.load_data()
    #     raw_copy: mne.io.RawArray = raw.copy()
    #
    #     raw_copy.apply_function(zscore)
    #     raw_copy.apply_function(robust_scale)
    #
    #     raw_copy.filter(0.3, 100, fir_design="firwin")
    #     raw_copy.notch_filter(freqs=np.arange(60, 121, 60))
    #     raw_copy.set_eeg_reference(ref_channels=["Ref"])
    #     raw_copy.drop_channels(["Ref"])
    #
    #     iter_freqs = [
    #         ("Beta", 13, 25),
    #         ("Gamma", 30, 45),
    #     ]
    #     freqs = [(13, 25), (30, 45)]
    #     coh_data = {}
    #
    #     tmin, tmax = -1, 0  # time window (seconds)
    #     chans = ["LFP1_vHp", "LFP4_AON"]
    #     for band, fmin, fmax in iter_freqs:
    #         band_path = session_path / band
    #         band_path.mkdir(parents=True, exist_ok=True)
    #
    #         raw_copy.pick_channels(chans)
    #         raw_copy.filter(
    #             fmin,
    #             fmax,
    #             n_jobs=None,
    #             l_trans_bandwidth=1,
    #             h_trans_bandwidth=1,
    #         )
    #         epoch = mne.Epochs(
    #             raw_copy,
    #             row["end"],
    #             row["event_id"],
    #             tmin,
    #             tmax,
    #             baseline=None,
    #             event_repeated="drop",
    #             preload=True,
    #         )
    #
    #         freqs_arange = np.arange(fmin, fmax)
    #         con = spectral_connectivity_epochs(
    #             epoch,
    #             indices=(np.array([0]), np.array([1])),
    #             method="coh",
    #             mode="cwt_morlet",
    #             cwt_freqs=freqs_arange,
    #             cwt_n_cycles=freqs_arange / 2,
    #             sfreq=epoch.info["sfreq"],
    #             n_jobs=1,
    #         )
    #         times = epoch.times
    #         coh_data = np.squeeze(con.get_data())
    #
    #         title = f"{band} Coherence {chans[0]} vs {chans[1]}"
    #         filename = f"{band}_{chans[0]}_{chans[1]}.png"
    #         plots.plot_2D_coherence(
    #             coh_data,
    #             times,
    #             np.arange(fmin, fmax),
    #             title,
    #             filename,
    #             band_path,
    #         )
    #         plot_epoch_data(epoch)
