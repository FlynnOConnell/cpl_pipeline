from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import mne_connectivity
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import decimate, butter, filtfilt, medfilt

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5

# matplotlib.use("TkAgg")

logger.setLevel("INFO")
sns.set_style("darkgrid")

# Set up 'ggplot' style
# plt.style.use("ggplot")

# # Configure parameters for publication-ready graphs
# plt.rcParams["axes.labelsize"] = 20  # Label size
# plt.rcParams["axes.titlesize"] = 22  # Title size
# plt.rcParams["xtick.labelsize"] = 14  # x-axis tick label size
# plt.rcParams["ytick.labelsize"] = 14  # y-axis tick label size
# plt.rcParams["legend.fontsize"] = 16  # Legend font size
# plt.rcParams["lines.linewidth"] = 2  # Line width
# plt.rcParams["axes.titleweight"] = "bold"  # Title weight
# plt.rcParams["axes.labelweight"] = "bold"  # Label weight
# plt.rcParams["axes.spines.top"] = False  # Remove top border
# plt.rcParams["axes.spines.right"] = False  # Remove right border
# plt.rcParams["axes.spines.left"] = True
# plt.rcParams["axes.spines.bottom"] = True
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["text.color"] = "black"


def get_data(path: Path | str):
    path = Path(path)
    if path.is_file():
        h5_file = spk_h5.read_h5(path)
    else:
        files = list(path.glob("*dk3*"))
        file = files[0]
        h5_file = spk_h5.read_h5(path / file)
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


# spikes - raw data
# times - raw data
# events - events array
# event_times - event times array
# fs - sampling frequency

# This class' goals:
# 1. Apply filters to the raw data
# 2. Calculate FFT and Frequencies


def clip_large_amplitudes(data, n_std_dev: float | int = 5):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    upper_threshold = mean_val + n_std_dev * std_dev
    lower_threshold = mean_val - n_std_dev * std_dev
    return np.clip(data, lower_threshold, upper_threshold)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y


class LfpSignal:
    def __init__(
        self,
        spikes_df,
        fs,
        ev_spikes_arr=None,
        ev_times_arr=None,
        ev_id_dict=None,
        ev_min_dur=None,
        window=(),
        filename=None,
        exclude=(),
    ):
        self._exclude = exclude
        self.filename = filename
        self.fs = fs
        self.events = ev_spikes_arr
        self.ev_times = ev_times_arr
        self.ev_id_dict = ev_id_dict
        self.min_duration = ev_min_dur
        self.spikes: pd.DataFrame = spikes_df.drop(columns=exclude, errors="ignore")
        self._channels = list(self.spikes.columns)
        spikes_arr = self.spikes.to_numpy().T
        ch_names = list(self.spikes.columns)
        ch_types = ["eeg"] * len(ch_names)

        # mne objects
        self._tmin = window[0] if len(window) > 0 else 0
        self._tmax = window[1] if len(window) > 1 else 1
        self.info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        self.raw = mne.io.RawArray(spikes_arr, self.info)
        self.filtered = False
        self.notched = False

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
        self.tmax = self.tmax if self.tmax is not None else self.min_duration
        return mne.Epochs(
            self.raw,
            events=self.events,
            event_id=self.ev_id_dict,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            picks=self.channels,
            detrend=1,
            preload=True,
        )

    def filter(self, inplace=True, **kwargs):
        if inplace:
            self.raw.filter(**kwargs)
            self.filtered = True
        else:
            return self.raw.copy().filter(**kwargs)

    def notch_filter_raw(self, freqs: list | np.ndarray = None, inplace=True):
        if freqs is None:
            freqs = np.arange(60, 121, 60)
        if inplace:
            self.raw.notch_filter(freqs=freqs)
            self.notched = True
        else:
            return self.raw.copy().notch_filter(freqs=freqs)


def plot_coh(epoch_object: mne.Epochs):
    # Connectivity Analysis
    indices = (np.array([0]), np.array([1]))  # Channel indices for connectivity
    freqs = np.arange(5, 100)
    n_cycles = freqs / 2

    con = mne_connectivity.spectral_connectivity_epochs(
        epoch_object,
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=n_cycles,
        sfreq=epoch_object.info["sfreq"],
        n_jobs=1,
    )
    times = epoch_object.times
    coh = con.get_data()  # The connectivity matrix, shape will be (n_freqs, n_times)

    plt.imshow(
        np.squeeze(coh),
        extent=(times[0], times[-1], freqs[0], freqs[-1]),
        aspect="auto",
        origin="lower",
        cmap="jet",
    )
    title = f"{epoch_object.ch_names[0]} vs {epoch_object.ch_names[1]}"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def process_events(events: np.ndarray, times: np.ndarray, fs=2000):
    processed = []
    min_dur = np.inf
    ev_dict = {}

    start_time, end_time = None, None
    for i in range(0, len(events)):
        if events[i].isalpha():
            start_time = times[i]
            j = i + 1

            # loops through to the last digit in sequence
            while j < len(events) and not events[j].isalpha():
                end_time = times[j]
                j += 1

            # keep an active minimum duration for proper epochs
            duration = end_time - start_time
            if duration < min_dur:
                min_dur = duration

            event_label = f"{events[i]}_{events[j - 1]}"
            if event_label not in ev_dict:
                ev_dict[event_label] = len(ev_dict) + 1
            event_id = ev_dict[event_label]
            processed.append([int(end_time * fs), 0, event_id])
    return processed, ev_dict, min_dur


def median_filter(data):
    return medfilt(data, kernel_size=5)  # Adjust kernel_size as needed


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"
    save_path = Path().home() / "data" / "figures"
    file = list(data_path.glob("*0609*.h5"))[0]
    df_s, df_t, ev, event_times = get_data(file)
    ev, event_id_dict, min_duration = process_events(ev, event_times)

    lfp = LfpSignal(
        df_s,
        2000,
        ev_spikes_arr=ev,
        ev_times_arr=event_times,
        filename=file,
        exclude=["LFP1_AON", "LFP2_AON"],
        ev_id_dict=event_id_dict,
        ev_min_dur=min_duration,
    )

    lfp.tmin = -0.5
    lfp.tmax = 0.5
    lfp.raw.filter(0.3, 100, fir_design="firwin")
    lfp.raw.notch_filter(freqs=np.arange(60, 121, 60))
    raw = lfp.raw.copy()

    chans = ["LFP1_vHp", "LFP3_AON"]
    epochs: mne.Epochs = lfp.epochs

    epochs.pick(chans)
    epochs['b_1'].compute_psd(method='welch').plot()
    epochs['w_1'].compute_psd(method='welch').plot()

    x = 2
