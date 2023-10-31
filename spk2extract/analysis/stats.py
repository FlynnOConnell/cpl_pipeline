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
        print(event)
        if event.isalpha():  # a-z
            print(f"{event} is alpha")

            # We know that the event is a letter, so we can check if:
            #   1) start time means this isn't the first letter event
            #   2) there is no previous event, or
            #   3) the previous event was a digit
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
        )
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
                windows.append(window)
            start_time, start_event = times[i], event
        else:
            end_time = times[i]
    # Handle the last interval
    if start_time is not None and end_time is not None:
        interval = f"{start_event}_{events[-1]}"
        event_id = id_dict.setdefault(interval, event_id_counter)
        window = (int(start_time * 1000), int(end_time * 1000), event_id)
        windows.append(window)
    return windows, id_dict


class LfpSignal:
    def __init__(
        self,
        data,
        fs,
        chan_names=None,
        events=None,
        event_id=None,
        filename=None,
    ):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy().T  # Channels x Time points
            ch_names = list(data.columns)
        elif isinstance(data, np.ndarray):
            self.data = data
            if chan_names is None:
                logger.warn("No channel names provided, generating channel names.")
                ch_names = [f"Ch_{i}" for i in range(self.data.shape[0])]
            else:
                ch_names = chan_names
        else:
            raise ValueError("Data must be a pandas DataFrame or numpy array.")

        self.filename = filename
        self.fs = fs
        self.events = events
        self.event_id = event_id
        self._channels = ch_names

        ch_types = ["eeg"] * len(ch_names)

        # mne objects
        self._tmin = -0.5
        self._tmax = 0.5
        self.info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        self.raw = mne.io.RawArray(self.data, self.info)
        self.filtered = False
        self.notched = False

    def resample(self, fs):
        self.fs = fs
        self.raw.resample(fs)

    @property
    def tmin(self):
        return self._tmin

    @tmin.setter
    def tmin(self, value):
        self._tmin = value

    @property
    def tmax(self):
        return self._tmax

    @tmax.setter
    def tmax(self, value):
        self._tmax = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def epochs(self):
        self.tmax = self.tmax if self.tmax is not None else 1
        return mne.Epochs(
            self.raw,
            events=self.events,
            event_id=self.event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            picks=self.channels,
            detrend=1,
            preload=True,
        )


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"

    save_path = Path().home() / "data" / "figures"
    save_path.mkdir(exist_ok=True)
    filelist = list(data_path.glob("*.h5"))
    errorfiles = []
    all_event_stats = pd.DataFrame()
    animals = list(data_path.iterdir())
    for animal_path in animals:
        animal = animal_path.name
        filelist = list(animal_path.glob("*.h5"))

        for file in filelist:
            h5 = spk_h5.read_h5(file)
            events_list, signals_list, other = [], [], []
            exclude = ["LFP1_AON", "LFP2_AON"]
            metadata = h5["channels"]["metadata"]
            respiratory = None
            for chan, item in h5["channels"].items():
                if chan not in exclude:
                    if "type" in item:
                        if item["type"] == "event":
                            tup = (chan, item["data"], item["times"])
                            events_list.append(tup)
                        elif item["type"] == "signal":
                            if chan in ['respirat', 'Respirat']:
                                respiratory = (chan, item["data"], item["times"], item["metadata"])
                            else:
                                tup = (chan, item["data"], item["times"], item["metadata"])
                                signals_list.append(tup)

            padded = pad_arrays_to_same_length([item[1] for item in signals_list])  # only for reference channel
            spikes_arr = np.vstack(padded)
            chans = [item[0] for item in signals_list]
            events_windows, ev_id_dict = process_event_windows(
                events_list[0][1], events_list[0][2]
            )

            windows = [
                (start / 1000, end / 1000, ev_id)
                for start, end, ev_id in events_windows
            ]
            df = pd.DataFrame(windows, columns=["Start", "End", "Event_ID"])

            df["Duration"] = df["End"] - df["Start"]
            event_stats = (
                df.groupby("Event_ID")
                .agg(
                    num_events=("Start", "count"),
                    avg_duration=("Duration", "mean"),
                    median_duration=("Duration", "median"),
                    min_duration=("Duration", "min"),
                    max_duration=("Duration", "max"),
                    std_duration=("Duration", "std"),
                )
                .reset_index()
            )

            event_stats["Event_Name"] = event_stats["Event_ID"].map(
                {v: k for k, v in ev_id_dict.items()}
            )

            event_stats["Animal"] = animal  # Add the animal identifier
            event_stats["File"] = file.name  # Add the file name

            # Append to the overall DataFrame
            all_event_stats = pd.concat(
                [all_event_stats, event_stats], ignore_index=True
            )

    aes = all_event_stats.copy()
    aes = aes[aes["num_events"] > 1]

