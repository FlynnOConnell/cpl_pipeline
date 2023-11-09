from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import numpy as np
import pandas as pd

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5

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
    for chan_idx, data in spikedata.items():
        if chan_idx in ["VERSION", "CLASS", "TITLE"]:
            continue
        spikes = data["spikes"]
        times = data["times"]
        spikes_df[chan_idx] = spikes
        times_df[chan_idx] = times
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

def process_event_windows_og(events: np.ndarray, times: np.ndarray):
    windows = []
    id_dict = {}
    event_id_counter = 1
    start_time, start_event, end_time = None, None, None
    for i, event in enumerate(events):
        if event == 'O':  # Check for uppercase 'O' to start interval
            start_time, start_event = times[i], event
        elif event == 'o':  # Check for lowercase 'o' to end interval
            if start_time is not None:
                end_time = times[i]
                interval = f"{start_event}_{event}"  # This will always be "O_o"
                event_id = id_dict.setdefault(interval, event_id_counter)
                if event_id == event_id_counter:
                    event_id_counter += 1
                window = (int(start_time * 1000), int(end_time * 1000), event_id)
                windows.append(np.array(window))
                # Reset the start_time for the next interval
                start_time, start_event = None, None

    return windows, id_dict

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

def get_first_event_time(events, times):
    for i, event in enumerate(events):
        if event.isalpha():
            return times[i]


if __name__ == "__main__":

    # data_path =  Path("/Volumes/CaTransfer/extracted")  # external hard drive, nfst
    data_path = Path().home() / "data" / 'extracted' / "serotonin"
    cache_path = Path().home() / "data" / ".cache" / 'serotonin'
    errorfiles = []
    all_event_stats = pd.DataFrame()
    animals = [p for p in data_path.iterdir() if p.is_dir()]

    for animal_path in animals:

        cache_animal_path = cache_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        animal = animal_path.name

        for file in animal_path.glob("*.h5"):

            session_name = file.stem
            print(f"Processing {session_name}")

            if cache_animal_path.joinpath(file.name).exists():
                print(f"Not (but we should be) skipping {file.name}")
                # TODO: check if the file is complete
                continue

            h5 = spk_h5.read_h5(file)
            events_list, signals_list, other = [], [], []
            exclude = [
                # "LFP1_AON",
                # "LFP2_AON",
            ]
            metadata = h5["channels"]["metadata"]

            lfp_signals_list = []
            unit_signals_list = []
            other_signals_list = []

            respiratory = None
            sniff = None
            reference = None

            exclude += ["VERSION", "CLASS", "TITLE", "metadata"]
            for chan, item in h5["channels"].items():
                print(f"Processing {chan}")
                if chan in exclude:
                    continue
                if hasattr(item, "items"):
                    if "type" in item.keys():
                        if item["type"] == "event":
                            events_list.append((chan, item["data"], item["times"]))
                        elif item["type"] == "signal":
                            if chan in ['respirat', 'Respirat']:
                                respiratory = (chan, item["data"], item["times"], item["metadata"])
                            elif chan in ['sniff', 'Sniff', 'snif', "sniff"]:
                                sniff = (chan, item["data"], item["times"], item["metadata"])
                            elif chan in ['ref', 'Ref', 'REF', 'refbrain', 'RefBrain', 'refbrain1', 'RefBrain1']:
                                reference = (chan, item["data"], item["times"], item["metadata"])
                            elif 'lfp' in chan.lower():
                                lfp_signals_list.append((chan, item["data"], item["times"], item["metadata"]))
                            elif 'u' in chan.lower():
                                try:
                                    unit_signals_list.append((chan, item["data"], item["times"], item["metadata"]))
                                except:
                                    logger.debug(f"Could not process {chan}")
                                    continue
                            else:
                                other_signals_list.append((chan, item["data"], item["times"], item["metadata"]))

            # Process and save LFP signals
            lfp_padded = [item[1] for item in lfp_signals_list]
            lfp_spikes_arr = np.vstack(lfp_padded) if lfp_padded else None
            lfp_ch_names = [item[0] for item in lfp_signals_list]
            lfp_metadata = [item[3] for item in lfp_signals_list]

            lfp_fs = [item["fs"] for item in lfp_metadata]

            # Process and save unit signals
            unit_padded = [item[1] for item in unit_signals_list]
            unit_spikes_arr = np.vstack(unit_padded) if unit_padded else None
            unit_ch_names = [item[0] for item in unit_signals_list]
            unit_metadata = [item[3] for item in unit_signals_list]

            unit_fs = [item["fs"] for item in unit_metadata]

            # Process and save other signals
            if other_signals_list:
                other_padded = [item[1] for item in other_signals_list]
                other_spikes_arr = np.vstack(other_padded) if other_padded else None
                other_ch_names = [item[0] for item in other_signals_list]
                other_metadata = [item[3] for item in other_signals_list]

                other_fs = [item["fs"] for item in other_metadata]

            # Process and save events
            fif_savename = cache_animal_path.joinpath(session_name)
            if lfp_spikes_arr is not None:

                freq = np.unique(lfp_fs)
                lfp_info = mne.create_info(
                    ch_names=lfp_ch_names,
                    sfreq=lfp_fs[0],  # Replace with actual sampling frequency if necessary
                    ch_types=["eeg"] * len(lfp_ch_names),)
                lfp_raw = mne.io.RawArray(lfp_spikes_arr, lfp_info)
                lfp_raw.save(fif_savename.with_name(f"{session_name}_lfp_raw.fif"), overwrite=True)

            if unit_spikes_arr is not None:

                freq = np.unique(unit_fs)
                unit_info = mne.create_info(
                    ch_names=unit_ch_names,
                    sfreq=lfp_fs[0],
                    ch_types=["eeg"] * len(unit_ch_names),
                )
                unit_raw = mne.io.RawArray(unit_spikes_arr, unit_info)
                unit_raw.save(fif_savename.with_name(f"{session_name}_unit_raw.fif"), overwrite=True)

            if respiratory is not None:
                np.save(str(cache_animal_path.joinpath(session_name + "_respiratory_signal")), respiratory[1])
                np.save(str(cache_animal_path.joinpath(session_name + "_respiratory_times")), respiratory[2])

            if sniff is not None:
                np.save(str(cache_animal_path.joinpath(session_name + "_sniff_signal")), sniff[1])
                np.save(str(cache_animal_path.joinpath(session_name + "_sniff_times")), sniff[2])

            if reference is not None:
                np.save(str(cache_animal_path.joinpath(session_name + "_reference_signal")), reference[1])
                np.save(str(cache_animal_path.joinpath(session_name + "_reference_times")), reference[2])

            if other_signals_list:
                other_info = mne.create_info(
                    ch_names=other_ch_names,
                    sfreq=lfp_fs[0],  # Replace with actual sampling frequency if necessary
                    ch_types=["eeg"] * len(other_ch_names),  # Update this to 'unit' or appropriate type
                )
                other_raw = mne.io.RawArray(other_spikes_arr, other_info)
                # Save Unit RawArray to file
                other_raw.save(fif_savename.with_name(f"{session_name}_other_raw.fif"), overwrite=True)

            ev_savename = cache_animal_path.joinpath(session_name + "_eve")
            ev_dict_savename = cache_animal_path.joinpath(session_name + "_id_ev")

            events_mne, ev_id_dict = process_event_windows_og(
                events_list[0][1], events_list[0][2]
            )

            events_mne_np = np.array(events_mne)
            keys_np = np.array(list(ev_id_dict.keys()), dtype=object)
            vals_np = np.array(list(ev_id_dict.values()), dtype=int)

            np.savez(
                ev_dict_savename.with_suffix(".npz"), keys=keys_np, values=vals_np
            )
            mne.write_events(
                ev_savename.with_suffix(".fif"), np.array(events_mne), overwrite=True
            )
    x = 2
