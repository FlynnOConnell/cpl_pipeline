from __future__ import annotations

import re
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd

from spk2extract.logs import logger

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

    # return mne.Epochs(
    #     self.raw,
    #     events=self.events,
    #     event_id=self.event_id,
    #     tmin=self.tmin,
    #     tmax=self.tmax,
    #     baseline=None,
    #     picks=self.channels,
    #     detrend=1,
    # )


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


if __name__ == "__main__":
    cache_path = Path().home() / "data" / ".cache"
    errorfiles = []
    animals = list(cache_path.iterdir())
    for animal_path in animals:  # each animal has a folder
        animal_data = []
        cache_animal_path = cache_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        animal = animal_path.name
        raw_files = list(animal_path.glob("*_raw*.fif"))
        event_files = list(animal_path.glob("*_eve*.fif"))

        metadata = [extract_file_info(raw_file.name) for raw_file in raw_files]
        raw_data = [mne.io.read_raw_fif(raw_file) for raw_file in raw_files]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        all_data_df = pd.concat(metadata)

        ev_start_holder = []
        ev_end_holder = []
        for arr in event_data:
            start_events = np.column_stack(
                [arr[:, 0], np.zeros(arr.shape[0]), arr[:, 2]]
            ).astype(int)
            end_events = np.column_stack([arr[:, 1], np.zeros(arr.shape[0]), arr[:, 2]]).astype(int)
            ev_start_holder.append(start_events)
            ev_end_holder.append(end_events)

        all_data_df["Raw"] = raw_data
        all_data_df["Events"] = event_data

        x = 2

    #     spikes_arr,
    #     2000,
    #     chan_names=chans,
    #     events=events_mne,
    #     event_id=ev_id_dict,
    #     filename=file,
    # )
    # lfp.tmin = -1
    # lfp.tmax = 0
    #
    # lfp.resample(1000)
    # lfp.raw.filter(0.3, 100, fir_design="firwin")
    # lfp.raw.notch_filter(freqs=np.arange(60, 121, 60))
    # lfp.raw.set_eeg_reference(ref_channels=["Ref"])
    #
    # no_ref_chans = ("LFP1_vHp", "LFP2_vHp", "LFP3_AON", "LFP4_AON")
    # groups = (
    #     ("LFP1_vHp", "LFP3_AON"),
    #     ("LFP1_vHp", "LFP4_AON"),
    #     ("LFP2_vHp", "LFP3_AON"),
    #     ("LFP2_vHp", "LFP4_AON"),
    # )
    #
    # lfp.nochans = lfp.raw.copy().pick(no_ref_chans)
    #
    # # beta frequencies
    # freqs = np.arange(13, 30)
    # # plots.plot_custom_data(lfp.nochans.get_data(), lfp.nochans.times, no_ref_chans, 200, 1, 1000)
    # for group in groups:
    #     epochs: mne.Epochs = lfp.epochs.copy()
    #     # plots.plot_coh(epochs.pick(group))
    #     beta = epochs.copy().pick(group)
    #     beta = beta.filter(freqs[0], freqs[-1])
    #     plots.plot_coh(beta, freqs=freqs)
