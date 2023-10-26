from __future__ import annotations
from itertools import combinations
from collections import defaultdict
from mne_connectivity import spectral_connectivity_epochs
from sklearn.manifold import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel
from scipy.signal import coherence

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval

from spk2extract.logs import logger
from spk2extract.viz import plots, helpers


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


if __name__ == "__main__":
    plots_path = Path().home() / "data" / "plots" / "aon"
    plots_path.mkdir(parents=True, exist_ok=True)
    data = get_master_df()
    data.reset_index(drop=True, inplace=True)
    num_sessions = data.shape[0]
    # get the timestamps when for each key in the event_id dict
    for i in range(num_sessions):
        animal_path = plots_path / f"{data.iloc[i]['Animal']}"
        session_path = (
            animal_path
            / f"{data.iloc[i]['Date']}_{data.iloc[i]['Day']}_{data.iloc[i]['Stimset']}_{data.iloc[i]['Context']}"
        )
        session_path.mkdir(parents=True, exist_ok=True)

        row = data.iloc[i, :]
        id_dict = row["event_id"]
        raw = row["raw"]
        start = row["start"]
        end = row["end"]
        start_time = start[0, 0] / 1000

        raw.load_data()
        raw_copy = raw.copy()
        raw_copy.filter(0.3, 100, fir_design="firwin")
        raw.notch_filter(freqs=np.arange(60, 121, 60))
        raw_copy.set_eeg_reference(ref_channels=["Ref"])
        raw_copy.drop_channels(["Ref"])

        def order_func(times, data):
            this_data = data[:, (times > -0.350) & (times < 0.0)]
            this_data /= np.sqrt(np.sum(this_data**2, axis=1))[:, np.newaxis]
            return np.argsort(
                spectral_embedding(
                    rbf_kernel(this_data, gamma=1.0), n_components=1, random_state=0
                ).ravel()
            )

        tmin, tmax = -0.5, 0
        epochs = mne.Epochs(
            raw_copy,
            row["end"],
            row["event_id"],
            tmin,
            tmax,
            baseline=None,
            event_repeated="drop",
            preload=True,
        )

        iter_freqs = [
            ("Beta", 13, 25),
            ("Gamma", 30, 45),
        ]
        freqs = [(13, 25), (30, 45)]

        frequency_map = list()
        coh_data = {}
        freq_data = {}
        times_data = {}

        tmin, tmax = -1, 0  # time window (in seconds)
        chans = ["LFP1_vHp", "LFP4_AON"]
        for band, fmin, fmax in iter_freqs:
            band_path = session_path / band
            band_path.mkdir(parents=True, exist_ok=True)

            raw_copy.pick_channels(chans)
            raw_copy.filter(
                fmin,
                fmax,
                n_jobs=None,  # use more jobs to speed up.
                l_trans_bandwidth=1,  # make sure filter params are the same
                h_trans_bandwidth=1,
            )
            epoch = mne.Epochs(
                raw_copy,
                row["end"],
                row["event_id"],
                tmin,
                tmax,
                baseline=None,
                event_repeated="drop",
                preload=True,
            )

            freqs = np.arange(fmin, fmax)
            con = spectral_connectivity_epochs(
                epoch,
                indices=(np.array([0]), np.array([1])),
                method="coh",
                mode="cwt_morlet",
                cwt_freqs=freqs,
                cwt_n_cycles=freqs / 2,
                sfreq=epoch.info["sfreq"],
                n_jobs=1,
            )
            times = epoch.times
            coh_data = np.squeeze(con.get_data())

            title = f"{band} Coherence {chans[0]} vs {chans[1]}"
            filename = f"{band}_{chans[0]}_{chans[1]}.png"
            plots.plot_2D_coherence(
                coh_data,
                times,
                freqs,
                title,
                filename,
                session_path,
            )
            stored_signals = defaultdict(lambda: defaultdict(list))
            for epoch_num, data in enumerate(epoch.get_data()):
                for epoch_num, data in enumerate(epoch.get_data()):
                    for chan_idx, chan_name in enumerate(epoch.ch_names):
                        signal = data[chan_idx, :]
                        stored_signals[chan_name][band].append(signal)

            highest_coh = 0
            best_pair = None
            best_band = None
            best_coh_freq = None

            sfreq = epoch.info["sfreq"]
            freqs = np.fft.fftfreq(len(epoch.times), 1 / sfreq)

            # Go through every combination of channels and bands
            for chan1, chan2 in combinations(stored_signals.keys(), 2):
                for band in iter_freqs:
                    band_name, fmin, fmax = band
                    signals1 = np.array(
                        stored_signals[chan1][band_name]
                    )  # Assuming trials x time
                    signals2 = np.array(stored_signals[chan2][band_name])

                    avg_signal1 = signals1.mean(axis=0)
                    avg_signal2 = signals2.mean(axis=0)

                    f, coh_values = coherence(avg_signal1, avg_signal2, fs=sfreq)

                    # Restrict to the band
                    indices = np.where((f >= fmin) & (f <= fmax))
                    max_coh_value = np.max(coh_values[indices])
                    max_coh_freq = f[np.argmax(coh_values[indices])]

                    if max_coh_value > highest_coh:
                        highest_coh = max_coh_value
                        best_pair = (chan1, chan2)
                        best_band = band_name
                        best_coh_freq = max_coh_freq

            plt.figure()
            plt.plot(
                epoch.times,
                np.mean(stored_signals[best_pair[0]][best_band], axis=0),
                label=f"{best_pair[0]}",
            )
            plt.plot(
                epoch.times,
                np.mean(stored_signals[best_pair[1]][best_band], axis=0),
                label=f"{best_pair[1]}",
            )
            plt.title(
                f"Highest Coherence between {best_pair[0]} and {best_pair[1]} at {best_coh_freq} Hz in {best_band} band"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.show()
            # get analytic signal (envelope)
            epochs.apply_hilbert(envelope=True)
            frequency_map.append(((band, fmin, fmax), epochs.average()))
        x = 4
