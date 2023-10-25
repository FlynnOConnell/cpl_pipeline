from __future__ import annotations
from sklearn.manifold import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel
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


if __name__ == "__main__":
    data = get_master_df()
    data.reset_index(drop=True, inplace=True)
    num_sessions = data.shape[0]
    # get the timestamps when for each key in the event_id dict
    for i in range(num_sessions):
        row = data.iloc[i, :]
        id_dict = row["event_id"]
        raw = row["raw"]
        start = row["start"]
        end = row["end"]
        start_time = start[0, 0] / 1000

        raw.load_data()

        raw.filter(0.3, 100, fir_design="firwin")
        raw.notch_filter(freqs=np.arange(60, 121, 60))
        raw.set_eeg_reference(ref_channels=["Ref"])
        raw.drop_channels(["Ref"])

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
            raw,
            row["end"],
            row["event_id"],
            tmin,
            tmax,
            baseline=None,
            event_repeated="drop",
            preload=True,
        )

        iter_freqs = [
            ("Theta", 4, 7),
            ("Alpha", 8, 12),
            ("Beta", 13, 25),
            ("Gamma", 30, 45),
        ]

        frequency_map = list()
        tmin, tmax = -1, 1  # time window (in seconds)
        for band, fmin, fmax in iter_freqs:
            raw = raw.copy().load_data()
            raw.filter(
                fmin,
                fmax,
                n_jobs=None,  # use more jobs to speed up.
                l_trans_bandwidth=1,  # make sure filter params are the same
                h_trans_bandwidth=1,
            )
            plots.plot_coh(raw)

            # epochs = mne.Epochs(
            #     raw,
            #     row["end"],
            #     row["event_id"],
            #     tmin,
            #     tmax,
            #     baseline=None,
            #     event_repeated="drop",
            #     preload=True,
            # )
            # epochs.subtract_evoked()
            #
            # # get analytic signal (envelope)
            # epochs.apply_hilbert(envelope=True)
            # frequency_map.append(((band, fmin, fmax), epochs.average()))

        x = 4