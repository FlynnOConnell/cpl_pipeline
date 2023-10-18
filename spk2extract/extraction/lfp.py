from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Generator

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import decimate, butter, filtfilt

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5

# matplotlib.use("TkAgg")

logger.setLevel("INFO")
sns.set_style("darkgrid")


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
        self.events = event_arr
        self.event_times = ev_times_arr
        self.fs = fs

        self.spikes: pd.DataFrame = spikes_df.drop(columns=exclude, errors="ignore")
        spikes_arr = self.spikes.to_numpy().T
        ch_names = list(self.spikes.columns)
        ch_types = ["eeg"] * len(ch_names)
        self.info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        self.raw = mne.io.RawArray(spikes_arr, self.info)

    def filter(self, low, high):
        self.raw.filter(low, high, fir_design="firwin")

    def notch(self):
        self.raw.notch_filter(60, fir_design="firwin")

    @property
    def custom_epochs(self, pre_time: float = 0.5, post_time: float = 0.5):
        return self.create_custom_epochs(pre_time, post_time)

    def create_custom_epochs(self, pre_time: float = 0.5, post_time: float = 0.5):
        epochs_data = []
        event_id = {}
        events = []
        for i, (letter, digit, epoch_idx, window) in enumerate(
            self.get_windows(pre_time, post_time, basis="end")
        ):
            # Clip large amplitudes

            epochs_data.append(window.to_numpy().T)

            event_key = f"{letter}_{digit}"
            if event_key not in event_id:
                event_id[event_key] = len(event_id) + 1
            events.append([i, 0, event_id[event_key]])

        events = np.array(events, dtype=int)

        return mne.EpochsArray(
            np.array(epochs_data),
            self.info,
            events=np.array(events, dtype=int),
            event_id=event_id,
        )

    def get_windows(
        self, pre_time: float = 0.5, post_time: float = 0.5, basis: str = "start"
    ) -> Generator:
        pre_time = abs(pre_time)
        post_time = abs(post_time)
        if not ensure_alternating(self.events):
            raise ValueError(
                "Events array does not alternate between letters and digits."
            )

        for i in range(0, len(self.events) - 1, 2):
            letter_str = self.events[i]
            digit_str = self.events[i + 1]
            if basis == "start":
                # use the letter as the basis, or the start of the interval
                epoch_idx = int(self.event_times[i] * self.fs)
                pre_time_idx = int(epoch_idx - (pre_time * self.fs))
                post_time_idx = int(epoch_idx + (post_time * self.fs))
            elif basis == "end":
                # use the digit as the basis, or the end of the interval
                epoch_idx = int(self.event_times[i + 1] * self.fs)
                pre_time_idx = int(epoch_idx - (pre_time * self.fs))
                post_time_idx = int(epoch_idx + (post_time * self.fs))
            elif basis == "all":
                pre_time_idx = int(self.event_times[i] * self.fs)
                post_time_idx = int(self.event_times[i + 1] * self.fs)
            else:
                raise ValueError(
                    f"Invalid basis {basis}. Must be one of 'start', 'end', or 'all'."
                )

            if not (0 <= pre_time_idx < len(self.spikes)) and (
                pre_time_idx < post_time_idx
            ):
                raise ValueError(
                    f"Pre time index {pre_time_idx} is out of bounds for the data."
                )

            if letter_str in ["b", "w"]:
                yield letter_str, digit_str, self.spikes.loc[pre_time_idx:post_time_idx]

    def remove_ica(self):
        ica = mne.preprocessing.ICA(
            n_components=len(self.spikes.columns), random_state=97, max_iter=800
        )
        ica.fit(self.raw)
        ica.plot_sources(self.raw)
        ica.apply(self.raw)


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"
    file = list(data_path.glob("*0609*.h5"))[0]
    df_s, df_t, ev, event_times = get_data(file)
    q = 2  # Decimation factor; 2000 Hz / 2 = 1000 Hz
    resampled_lfp_df = df_s.apply(lambda col: decimate(col, q), axis=0)

    lfp = LfpSignal(
        resampled_lfp_df,
        2000,
        event_arr=ev,
        ev_times_arr=event_times,
        filename=file,
        exclude=["LFP1_AON", "LFP2_AON"],
    )
    lfp.filter(15, 100)
    lfp.notch()

    x = 0
    channel_data = defaultdict(list)

    for letter_str, digit_str, spikes_window in lfp.get_windows(0.5, 0.5, "all"):
        x += 1
        if x > 10:
            break

        for channel in spikes_window.columns:
            channel_data[channel].append((letter_str, digit_str, spikes_window))

    for channels in [("LFP1_vHp", "LFP3_AON"), ("LFP2_vHp", "LFP4_AON")]:
        fig, ax = plt.subplots(2, 1, figsize=(10, 15))

        for i, channel in enumerate(channels):
            for letter_str, digit_str, spikes_window in channel_data[channel]:
                # Similar processing as your code
                idx_min, idx_max = spikes_window.index.min(), spikes_window.index.max()
                xticks = np.linspace(idx_min, idx_max, 11)
                xticklabels = np.linspace(0, (idx_max - idx_min) / 2000, 11).round(2)

                beta_filtered = butter_bandpass_filter(
                    spikes_window[channel], 12, 30, fs=2000
                )
                gamma_filtered = butter_bandpass_filter(
                    spikes_window[channel], 30, 100, fs=2000
                )

                ax[i].plot(
                    spikes_window.index,
                    spikes_window[channel],
                    label="Raw",
                    color="gray",
                    alpha=0.5,
                    zorder=1,
                )
                ax[i].plot(
                    spikes_window.index,
                    beta_filtered,
                    label="Beta",
                    color="r",
                    zorder=2,
                )
                ax[i].plot(
                    spikes_window.index,
                    gamma_filtered,
                    label="Gamma",
                    color="g",
                    zorder=3,
                )
                ax[i].set_title(f"{letter_str}_{digit_str} Channel: {channel}")
                ax[i].set_xticks(xticks)
                ax[i].set_xticklabels(xticklabels)
                ax[i].set_xlabel("Time (s)")
                ax[i].legend()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    x = 4
