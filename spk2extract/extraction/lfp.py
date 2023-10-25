from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import numpy as np
import pandas as pd

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5
from spk2extract.viz import plots

logger.setLevel("INFO")


def get_h5(datapath, match):
    # recursively find any files that match the pattern
    def find_files(fpath: Path, wildcard: str) -> Generator[Path, None, None]:
        return (matched_file for matched_file in fpath.glob(wildcard))

    datapath = Path(datapath)

    if datapath.is_dir():
        if match:
            files = list(find_files(datapath, match))
            logger.info(f"Found {len(files)} files matching {match}")
        else:
            files = list(
                datapath.glob("*")
            )  # default to all files if no match string is provided
            logger.info(f"Found {len(files)} files in {datapath}")
        if not files:
            raise FileNotFoundError(f"No files found in {datapath} matching {match}")
        datapath = files[0]

    if datapath.is_file():
        logger.info(f"Reading {datapath}")
        h5_file = spk_h5.read_h5(datapath)
    else:
        raise FileNotFoundError(f"Could not find file or directory {datapath}")
    return h5_file


def get_data(datapath: Path | str, match: str = None):
    # recursively find any files that match the pattern
    def find_files(fpath: Path, wildcard: str) -> Generator[Path, None, None]:
        return (matched_file for matched_file in fpath.glob(wildcard))

    datapath = Path(datapath)

    if datapath.is_dir():
        if match:
            files = list(find_files(datapath, match))
            logger.info(f"Found {len(files)} files matching {match}")
        else:
            files = list(
                datapath.glob("*")
            )  # default to all files if no match string is provided
            logger.info(f"Found {len(files)} files in {datapath}")
        if not files:
            raise FileNotFoundError(f"No files found in {datapath} matching {match}")
        datapath = files[0]

    if datapath.is_file():
        logger.info(f"Reading {datapath}")
        h5_file = spk_h5.read_h5(datapath)
    else:
        raise FileNotFoundError(f"Could not find file or directory {datapath}")

    spikes_df = pd.DataFrame()
    times_df = pd.DataFrame()

    events_arr = h5_file["events"]["events"]
    event_times_arr = h5_file["events"]["times"]

    spikedata = h5_file["spikedata"]
    for chan, data in spikedata.items():
        if chan in ["VERSION", "CLASS", "TITLE"]:
            continue
        spikes = data["spikes"]
        times = data["times"]
        spikes_df[chan] = spikes
        times_df[chan] = times
    return spikes_df, times_df, events_arr, event_times_arr


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


def process_events(events: np.ndarray, times: np.ndarray):
    ev_store = []
    id_dict = {}
    non_interval_events = []
    start_time, start_event, prev_event, end_time = None, None, None, None
    event_id_counter = 1

    for i, event in enumerate(events):
        if event.isalpha():  # a-z
            if start_time is not None and prev_event is not None:
                event_name = f"{start_event}_{prev_event}"
                event_id = id_dict.setdefault(event_name, event_id_counter)

                if event_id == event_id_counter:
                    event_id_counter += 1

                start_sample = int(
                    start_time
                )  # Converting to int and assuming times are in seconds
                ev_store.append([start_sample, 0, event_id])

                if end_time is not None:
                    non_interval_events.append(
                        int(end_time)
                    )  # Converting to int and assuming times are in seconds
            start_time, start_event = times[i], event
        else:
            end_time, prev_event = times[i], event

    if start_time is not None and prev_event is not None:
        event_name = f"{start_event}_{prev_event}"
        event_id = id_dict.setdefault(event_name, event_id_counter)
        start_sample = int(
            start_time
        )  # Converting to int and assuming times are in seconds
        ev_store.append([start_sample, 0, event_id])

        if end_time is not None:
            non_interval_events.append(int(end_time))

    ev_store = np.array(ev_store, dtype=int)
    non_interval_events = np.array(non_interval_events, dtype=int)
    return ev_store, id_dict, non_interval_events


def process_event_windows(events: np.ndarray, times: np.ndarray):
    windows = []
    id_dict = {}
    event_id_counter = 1
    start_time, start_event, end_time = None, None, None
    for i, event in enumerate(events):
        if event.isalpha():
            if start_time is not None and end_time is not None:
                interval = f"{start_event}_{events[i-1]}"
                event_id = id_dict.setdefault(interval, event_id_counter)
                if event_id == event_id_counter:
                    event_id_counter += 1
                window = (int(start_time * 1000), int(end_time * 1000), event_id)
                windows.append(np.array(window))
            start_time, start_event = times[i], event
        else:
            end_time = times[i]
    # Handle the last interval
    if start_time is not None and end_time is not None:
        interval = f"{start_event}_{events[-1]}"
        event_id = id_dict.setdefault(interval, event_id_counter)
        window = (int(start_time * 1000), int(end_time * 1000), event_id)
        windows.append(np.array(window))
    return windows, id_dict


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"
    cache_path = Path().home() / "data" / ".cache"
    filelist = list(data_path.glob("*.h5"))
    errorfiles = []
    all_event_stats = pd.DataFrame()
    animals = list(data_path.iterdir())

    for animal_path in animals:
        cache_animal_path = cache_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        animal = animal_path.name
        filelist = list(animal_path.glob("*.h5"))

        for file in filelist:
            session_name = file.stem
            # skip processed files
            if cache_animal_path.joinpath(file.name).exists():
                continue

            h5 = spk_h5.read_h5(file)
            events_list, signals_list, other = [], [], []
            exclude = [
                "LFP1_AON",
                "LFP2_AON",
                "Respirat",
            ]
            metadata = h5["channels"]["metadata"]

            exclude += ["VERSION", "CLASS", "TITLE", "metadata"]
            for chan, item in h5["channels"].items():
                if chan in exclude:
                    continue
                if hasattr(item, "items"):
                    if "type" in item.keys():
                        if item["type"] == "event":
                            tup = (chan, item["data"], item["times"])
                            events_list.append(tup)
                        # elif item["type"] == "signal":
                        #     tup = (chan, item["data"], item["times"], item["metadata"])
                        #     signals_list.append(tup)
                        # else:
                        #     other.append(item[chan])
            # padded = pad_arrays_to_same_length([item[1] for item in signals_list])
            # spikes_arr = np.vstack(padded)
            # ch_names = [item[0] for item in signals_list]
            # info = mne.create_info(
            #     ch_names=ch_names,
            #     sfreq=2000,
            #     ch_types=["eeg"] * len(ch_names),
            # )
            # raw = mne.io.RawArray(spikes_arr, info)
            # fif_savename = cache_animal_path.joinpath(session_name + "_raw")
            ev_savename = cache_animal_path.joinpath(session_name + "_eve")
            ev_dict_savename = cache_animal_path.joinpath(session_name + "_id_ev")
            #
            # raw.save(fif_savename.with_suffix(".fif"), overwrite=True)

            events_mne, ev_id_dict = process_event_windows(
                events_list[0][1], events_list[0][2]
            )
            events_mne_np = np.array(events_mne)
            keys_np = np.array(list(ev_id_dict.keys()), dtype=object)
            vals_np = np.array(list(ev_id_dict.values()), dtype=int)
            np.savez(
                ev_dict_savename.with_suffix(".npz"), keys=keys_np, values=vals_np
            )
            # mne.write_events(
            #     ev_savename.with_suffix(".fif"), np.array(events_mne), overwrite=True
            # )

            # lfp = LfpSignal(
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

    x = 2
