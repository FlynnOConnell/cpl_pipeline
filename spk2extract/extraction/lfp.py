from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import mne_connectivity
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import decimate, butter, filtfilt, medfilt

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5

# matplotlib.use("TkAgg")

logger.setLevel("INFO")
sns.set_style("darkgrid")

# Set up 'ggplot' style
plt.style.use("ggplot")

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
        event_arr=None,
        ev_times_arr=None,
        filename=None,
        exclude=(),
    ):
        self._exclude = exclude
        self.filename = filename

        # TODO: fix this monstrous hack
        self.events = event_arr
        self.event_times = ev_times_arr
        self.fs = fs

        self.spikes: pd.DataFrame = spikes_df.drop(columns=exclude, errors="ignore")
        spikes_arr = self.spikes.to_numpy().T
        ch_names = list(self.spikes.columns)
        ch_types = ["eeg"] * len(ch_names)
        self.info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        self.raw = mne.io.RawArray(spikes_arr, self.info)

    @property
    def epochs(self):
        events, event_id_dict = self.get_windowed_dict()
        return mne.Epochs(
            self.raw,
            events=events,
            event_id=event_id_dict,
            tmin=0,
            tmax=self.min_duration,
            baseline=None,
            picks=None,
            detrend=1,
        )

    def change_epoch_window(self, tmin, tmax, inplace=True):
        events, event_id_dict = self.get_windowed_dict()
        if inplace:
            mne.Epochs(
                self.raw,
                events=events,
                event_id=event_id_dict,
                tmin=0,
                tmax=self.min_duration,
                baseline=None,
                picks=None,
                detrend=1,
            )
        else:
            return self.epochs.copy().tmin(tmin).tmax(tmax)

    def filter(self, inplace=True, **kwargs):
        if inplace:
            self.raw.filter(**kwargs)
        else:
            return self.raw.copy().filter(**kwargs)

    def create_annotations(self):
        onsets = []
        durations = []
        descriptions = []

        for i in range(0, len(self.events)):
            onsets.append(self.event_times[i])
            if i < len(self.events) - 1:
                durations.append(self.event_times[i + 1] - self.event_times[i])
            else:
                durations.append(1)  # Or your default duration for the last event
            descriptions.append(self.events[i])

        annot = mne.Annotations(onsets, durations, descriptions)
        self.raw.set_annotations(annot)

    def get_windowed_dict(self):
        events = []
        event_id_dict = {}
        self.min_duration = float("inf")  # Initialize with a large value

        for i in range(0, len(self.events)):
            if self.events[i].isalpha():
                start_time = self.event_times[i]
                j = i + 1
                while j < len(self.events) and not self.events[j].isalpha():
                    end_time = self.event_times[j]
                    j += 1

                # Update the minimum duration
                duration = end_time - start_time
                if duration < self.min_duration:
                    min_duration = duration

                event_label = f"{self.events[i]}_{self.events[j - 1]}"
                if event_label not in event_id_dict:
                    event_id_dict[event_label] = len(event_id_dict) + 1

                event_id = event_id_dict[event_label]
                events.append([int(start_time * self.fs), 0, event_id])

        return events, event_id_dict

    def notch_filter_raw(self):
        self.raw.notch_filter(np.arange(60, 121, 60), picks="all")

    @staticmethod
    def clean_ica(epoch):
        from mne.preprocessing import ICA

        ica = ICA(n_components=len(epoch.ch_names), random_state=97, max_iter=800).fit(
            epoch
        )
        ica.exclude = [0, 1]
        return ica.apply(epoch.copy().crop(tmin=1, tmax=None), exclude=ica.exclude)


def graphify(data: np.ndarray):
    n_chan, n_sam, n_epoch = 1, 1, 1
    if len(data.shape) == 1:
        n_sam = data.shape
    if len(data.shape) == 2:
        n_chan, n_sam = data.shape
    if len(data.shape) == 3:
        n_epoch, n_chan, n_sam = data.shape

    fig, ax = plt.subplots(n_epoch, n_chan, sharex=True, sharey=True)
    xticks = np.arange(0, n_sam, 2000)
    xlabs = np.arange(0, n_sam / 2000, 1)
    for i in range(n_epoch):
        for j in range(n_chan):
            ax[i, j].plot(data[i, j, :])
    plt.show()
    return fig, ax


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
        indices=indices,
        sfreq=epoch_object.info["sfreq"],
        mt_adaptive=True,
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
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"
    save_path = Path().home() / "data" / "figures"
    file = list(data_path.glob("*0609*.h5"))[0]
    df_s, df_t, ev, event_times = get_data(file)

    lfp = LfpSignal(
        df_s,
        2000,
        event_arr=ev,
        ev_times_arr=event_times,
        filename=file,
        exclude=["LFP1_AON", "LFP2_AON"],
    )
    lfp.notch_filter_raw()

    epochs: mne.Epochs = lfp.epochs
    notched = epochs.copy().notch_filter(np.arange(60, 121, 60), picks="all")

    epochs.load_data()

    notch_freqs = np.array([60, 120], dtype=object)

    def median_filter(data):
        return medfilt(data, kernel_size=5)  # Adjust kernel_size as needed

    epochs_median = epochs.copy().apply_function(median_filter)

    x = 2
