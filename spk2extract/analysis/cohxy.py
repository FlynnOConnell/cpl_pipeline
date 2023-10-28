from __future__ import annotations

import re
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


def ensure_alternating(ev):
    if len(ev) % 2 != 0:
        return False, "List length should be even for alternating pattern."
    for i in range(0, len(ev), 2):
        if not ev[i].isalpha():
            return False, f"Expected a letter at index {i}, got {ev[i]}"
        if not ev[i + 1].isdigit():
            return False, f"Expected a digit at index {i+1}, got {ev[i + 1]}"
    return True, "List alternates correctly between letters and digits."


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


def gen_sessions(raw_obj: mne.io.RawArray, start, end):
    for s, e in zip(start, end):
        yield raw_obj.copy().crop(s, e)


def gen_epochs(raw_obj: mne.io.RawArray, start, end):
    for s, e in zip(start, end):
        yield raw_obj.copy().crop(s, e).make_fixed_length_events()


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


def get_master_df():
    cache_path = Path().home() / "data" / ".cache"
    animals = list(cache_path.iterdir())

    master = pd.DataFrame()
    for animal_path in animals:  # each animal has a folder
        cache_animal_path = cache_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(list(animal_path.glob("*_raw*.fif")), key=extract_common_key)
        event_files = sorted(
            list(animal_path.glob("*_eve*.fif")), key=extract_common_key
        )
        event_id = sorted(list(animal_path.glob("*_id_*.npz")), key=extract_common_key)
        animal = animal_path.name
        assert len(raw_files) == len(event_files) == len(event_id)

        metadata = [extract_file_info(raw_file.name) for raw_file in raw_files]
        raw_data = [mne.io.read_raw_fif(raw_file) for raw_file in raw_files]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        event_id_dicts = [read_npz_as_dict(id_file) for id_file in event_id]
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
        }
        metadata_df = pd.concat(metadata).reset_index(drop=True)
        all_data_df = pd.DataFrame(all_data_dict)
        all_data_df = pd.concat([metadata_df, all_data_df], axis=1)
        master = pd.concat([master, all_data_df], axis=0)
    return master


def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x**2, axis=0)


def iter_bands(freqs: tuple):
    if len(freqs) == 2:
        yield freqs
    else:
        for band, fmin, fmax in freqs:
            yield band, fmin, fmax


def order_func(times, data):
    this_data = data[:, (times > -0.350) & (times < 0.0)]
    this_data /= np.sqrt(np.sum(this_data**2, axis=1))[:, np.newaxis]
    return np.argsort(
        spectral_embedding(
            rbf_kernel(this_data, gamma=1.0), n_components=1, random_state=0
        ).ravel()
    )


def sliding_window_coherence(signal1, signal2, window_size, step_size, fs):
    n_points = len(signal1)
    start_indices = range(0, n_points - window_size, step_size)
    coherence_values = []

    for start in start_indices:
        end = start + window_size
        f, cxy = coherence(signal1[start:end], signal2[start:end], fs=fs)
        coherence_values.append(np.mean(cxy))
    return np.array(coherence_values)


# Calculate rolling coherence (replace with your actual method)
def rolling_coherence(x, y, window, fs=1.0):
    coh_vals = []
    for i in range(0, len(x) - window, window // 2):  # 50% overlap
        f, Cxy = coherence(
            x[i : i + window], y[i : i + window], fs=fs, nperseg=window // 2
        )
        coh_vals.append(
            np.mean(Cxy)
        )  # Average coherence across frequencies, adjust as needed
    return np.array(coh_vals)


def robust_scale(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    median = np.median(data)
    return (data - median) / iqr


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
                new_coh_times[seg : seg + 2],
                0,
                new_coh_values[seg : seg + 2],
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


def preprocess_raw(raw, chans=("LFP1_vHp", "LFP4_AON"), fmin=0.3, fmax=100):
    raw.load_data()
    raw.apply_function(zscore)
    raw.apply_function(robust_scale)
    raw.filter(fmin, fmax, fir_design="firwin")
    raw.notch_filter(freqs=np.arange(60, 121, 60))
    raw.set_eeg_reference(ref_channels=["Ref"])
    raw.drop_channels(["Ref"])
    raw.pick_channels(chans)
    return raw


def get_filtered_epoch(raw, events, event_id, fmin, fmax, tmin, tmax, baseline=None):
    raw.filter(fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1)
    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        baseline=baseline,
        event_repeated="drop",
    )


def process_session(raw_arr, events, event_id, fmin, fmax, tmin, tmax):
    assert fmin < fmax and tmin < tmax, "Ensure fmin < fmax and tmin < tmax"
    raw = preprocess_raw(raw_arr)
    valid_events, valid_event_id = _validate_events(events, event_id)
    epoch = get_filtered_epoch(
        raw, valid_events, valid_event_id, fmin, fmax, tmin, tmax
    )
    freqs_arange = np.arange(fmin, fmax)
    con = spectral_connectivity_epochs(
        epoch,
        indices=(np.array([0]), np.array([1])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs_arange,
        cwt_n_cycles=freqs_arange / 2,
        sfreq=epoch.info["sfreq"],
        n_jobs=1,
    )
    return epoch, con


def main():
    helpers.update_rcparams()
    plots_path = Path().home() / "data" / "plots" / "aon"
    plots_path.mkdir(parents=True, exist_ok=True)
    data = get_master_df()

    tmin, tmax = -1, 0  # time window (in seconds)
    fmin, fmax = 13, 25  # frequency range
    epoch_holder, con_arr_holder = [], []

    for idx, row in data.iterrows():
        epoch, con_arr = process_session(
            row["raw"], row["end"], row["event_id"], fmin, fmax, tmin, tmax
        )
        epoch_holder.append(epoch)
        con_arr_holder.append(con_arr)
    data["epoch"] = epoch_holder
    data["con_arr"] = con_arr_holder

    return data


def filter_data_by_conditions(df, **kwargs):
    # check if the kwargs are valid columns
    for key in kwargs.keys():
        if key not in df.columns:
            raise ValueError(f"{key} is not a valid column name")
    # filter the dataframe by rows that contain all the kwargs
    for key, value in kwargs.items():
        df = df[df[key] == value]
    return df


def compare_groups(df, group1_conditions, group2_conditions):
    group1_data = filter_data_by_conditions(df, **group1_conditions)
    group2_data = filter_data_by_conditions(df, **group2_conditions)


# Function for statistical comparison
def stats_comparison(epoch1, epoch2, ch_index=0, tmin=None, tmax=None):
    data1 = epoch1.get_data()[:, ch_index, :]
    data2 = epoch2.get_data()[:, ch_index, :]
    sample_times = epoch1.times
    if tmin and tmax:
        indices = (sample_times >= tmin) & (sample_times <= tmax)
        data1 = data1[:, indices]
        data2 = data2[:, indices]
    t_stat, p_val = ttest_rel(data1.mean(axis=0), data2.mean(axis=0))
    return t_stat, p_val

def optimal_nfft(n_times):
    return 2 ** np.ceil(np.log2(n_times))

def plot_psd(epochs1, epochs2, ch_names, fmin=1, fmax=100):

    data1 = epochs1.get_data(picks=ch_names)  # shape (n_epochs, n_channels, n_times)

    data1 = np.mean(data1, axis=0)  # Average over epochs
    n_fft = optimal_nfft(data1.shape[2])
    psds1, freqs1 = psd_array_welch(data1, epochs1.info['sfreq'], fmin=fmin, fmax=fmax, n_fft=n_fft)

    data2 = epochs2.get_data(picks=ch_names)
    data2 = np.mean(data2, axis=0)
    n_fft_2 = optimal_nfft(data2.shape[2])
    psds2, freqs2 = psd_array_welch(data2, epochs2.info['sfreq'], fmin=fmin, fmax=fmax, n_fft=n_fft_2)

    plt.figure()
    plt.loglog(freqs1, np.mean(psds1, axis=0))
    plt.loglog(freqs2, np.mean(psds2, axis=0))
    plt.title('Power Spectral Densities')
    plt.legend(['Condition 1', 'Condition 2'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.show()

def plot_coherence(epochs1, epochs2, ch1='AON', ch2='hippocampus', nperseg=1024):
    sfreq = epochs1.info['sfreq']
    f, Cxy1 = coherence(epochs1.get_data(picks=ch1).squeeze(), epochs1.get_data(picks=ch2).squeeze(), fs=sfreq, nperseg=nperseg)
    f, Cxy2 = coherence(epochs2.get_data(picks=ch1).squeeze(), epochs2.get_data(picks=ch2).squeeze(), fs=sfreq, nperseg=nperseg)

    plt.figure()
    plt.semilogx(f, Cxy1)
    plt.semilogx(f, Cxy2)
    plt.title(f'Coherence between {ch1} and {ch2}')
    plt.legend(['Condition 1', 'Condition 2'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.show()


if __name__ == "__main__":
    data = main()

    dk3 = data[data["Animal"] == "dk3"]
    dk3_context = dk3[dk3["Context"] == "context"]
    dk3_nocontext = dk3[dk3["Context"] == "nocontext"]

    epochs_context = dk3_context["epoch"].values
    epochs_nocontext = dk3_nocontext["epoch"].values
    key = {"b_1": 1, "b_0": 2, "w_1": 3, "w_0": 4, "x_1": 5, "x_0": 6}
    for e in epochs_context:
        e.event_id = key
    for e in epochs_nocontext:
        e.event_id = key
    con_concat = mne.concatenate_epochs(epochs_context.tolist())
    nocon_concat: mne.Epochs = mne.concatenate_epochs(epochs_nocontext.tolist())

    con_concat.plot_psd(fmin=5, fmax=70, average=True, )
    nocon_concat.plot_psd(fmin=5, fmax=70, average=True)
    t_stat, p_val = stats_comparison(con_concat, nocon_concat)
    print(f"T-statistic: {t_stat}, P-value: {p_val}")

    avg_con = con_concat.average()
    avg_nocon = nocon_concat.average()
    plot_psd(avg_con, avg_nocon, ['LFP1_vHp', 'LFP4_AON'])

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
