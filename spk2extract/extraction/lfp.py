from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import mne_connectivity
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import medfilt

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5
from spk2extract.extraction import plots

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
    events_mne = []
    ev_id_dict = {}
    in_between_events = []
    start_time, start_event, prev_event, end_time = None, None, None, None
    event_id_counter = 1

    for i, event in enumerate(events):
        if event.isalpha():
            if start_time is not None and prev_event is not None:
                interval = f"{start_event}_{prev_event}"
                event_id = ev_id_dict.setdefault(interval, event_id_counter)

                if event_id == event_id_counter:
                    event_id_counter += 1

                start_sample = int(
                    start_time * 1000
                )  # Converting to int and assuming times are in seconds
                events_mne.append([start_sample, 0, event_id])

                if end_time is not None:
                    in_between_events.append(
                        int(end_time * 1000)
                    )  # Converting to int and assuming times are in seconds

            start_time, start_event = times[i], event
        else:
            end_time, prev_event = times[i], event

    if start_time is not None and prev_event is not None:
        interval = f"{start_event}_{prev_event}"
        event_id = ev_id_dict.setdefault(interval, event_id_counter)
        start_sample = int(
            start_time * 1000
        )  # Converting to int and assuming times are in seconds
        events_mne.append([start_sample, 0, event_id])

        if end_time is not None:
            in_between_events.append(
                int(end_time * 1000)
            )  # Converting to int and assuming times are in seconds

    events_mne = np.array(events_mne, dtype=int)
    in_between_events = np.array(in_between_events, dtype=int)
    return events_mne, ev_id_dict, in_between_events

def median_filter(data):
    return medfilt(data, kernel_size=5)  # Adjust kernel_size as needed

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
    data_path = Path().home() / "data" / "extracted" / "dk1"
    save_path = Path().home() / "data" / "figures"
    save_path.mkdir(exist_ok=True)
    filelist = list(data_path.glob("*0609*.h5"))
    errorfiles = []
    file = filelist[0]
    h5 = get_h5(file, match="*.h5")
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
                elif item["type"] == "signal":
                    tup = (chan, item["data"], item["times"], item["metadata"])
                    signals_list.append(tup)
                else:
                    other.append(item[chan])
    padded = pad_arrays_to_same_length([item[1] for item in signals_list])
    spikes_arr = np.vstack(padded)
    chans = [item[0] for item in signals_list]
    events_mne, ev_id_dict, in_between_events = process_events(
        events_list[0][1], events_list[0][2]
    )

    lfp = LfpSignal(
        spikes_arr,
        2000,
        chan_names=chans,
        events=events_mne,
        event_id=ev_id_dict,
        filename=file,
    )
    lfp.tmin = -0.3
    lfp.tmax = 0.3

    plots.plot_custom_mne_raw(lfp.raw, "Raw", 100, 5, 1)
    lfp.resample(1000)
    plots.plot_custom_mne_raw(lfp.raw, "Resampled", 100, 5, 1)
    lfp.raw.filter(0.3, 100, fir_design="firwin")
    plots.plot_custom_mne_raw(lfp.raw, "Filtered", 100, 5, 1)
    lfp.raw.notch_filter(freqs=np.arange(60, 121, 60))
    plots.plot_custom_mne_raw(lfp.raw, "Notched", 100, 5, 1)
    lfp.raw.set_eeg_reference(ref_channels=["Ref"])
    plots.plot_custom_mne_raw(lfp.raw, "Referenced", 100, 5, 1)

    no_ref_chans = ("LFP1_vHp", "LFP2_vHp", "LFP3_AON", "LFP4_AON")
    groups = (
        ("LFP1_vHp", "LFP3_AON"),
        ("LFP1_vHp", "LFP4_AON"),
        ("LFP2_vHp", "LFP3_AON"),
        ("LFP2_vHp", "LFP4_AON"),
    )
    # for group in groups:
    #     epochs: mne.Epochs = lfp.epochs.copy()
    #     e = epochs.pick(group)
    #     plot_coh(e)

    x = 2
