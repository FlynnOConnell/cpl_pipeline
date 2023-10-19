from __future__ import annotations

from pathlib import Path
from typing import Generator

import mne
import mne_connectivity
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
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
    # Set up 'ggplot' style
    plt.style.use("ggplot")

    # Configure parameters for publication-ready graphs
    plt.rcParams["axes.labelsize"] = 20  # Label size
    plt.rcParams["axes.titlesize"] = 22  # Title size
    plt.rcParams["xtick.labelsize"] = 14  # x-axis tick label size
    plt.rcParams["ytick.labelsize"] = 14  # y-axis tick label size
    plt.rcParams["legend.fontsize"] = 16  # Legend font size
    plt.rcParams["lines.linewidth"] = 2  # Line width
    plt.rcParams["axes.titleweight"] = "bold"  # Title weight
    plt.rcParams["axes.labelweight"] = "bold"  # Label weight
    plt.rcParams["axes.spines.top"] = False  # Remove top border
    plt.rcParams["axes.spines.right"] = False  # Remove right border
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.color"] = "black"

    data_path = Path().home() / "data" / "extracted"
    save_path = Path().home() / "data" / "figures"
    file = list(data_path.glob("*0609*.h5"))[0]
    df_s, df_t, ev, event_times = get_data(file)
    # resampled to 1000Hz
    resampled_lfp_df = df_s.apply(lambda col: decimate(col, 2), axis=0)

    lfp = LfpSignal(
        resampled_lfp_df,
        1000,
        event_arr=ev,
        ev_times_arr=event_times,
        filename=file,
        exclude=["LFP1_AON", "LFP2_AON"],
    )
    lfp.filter(15, 100)
    lfp.notch()
    letter_map = {"b": "Black", "w": "White"}
    digit_map = {"0": "Incorrect dig", "1": "Correct dig"}

    data_by_trial_type = {}
    info = mne.create_info(
        ch_names=list(lfp.spikes.columns), sfreq=lfp.fs, ch_types="eeg"
    )
    data = []
    metadata_list = []

    for letter_str, digit_str, spikes_window in lfp.get_windows(0.5, 0.5, "all"):
        trial_key = f"{letter_str}_{digit_str}"
        if trial_key not in data_by_trial_type:
            data_by_trial_type[trial_key] = {}
        this_window = {}
        channel_pairs = [("LFP1_vHp", "LFP3_AON"), ("LFP2_vHp", "LFP4_AON")]
        for ch1, ch2 in channel_pairs:
            for channel in [ch1, ch2]:
                channel_data = {
                    "raw": spikes_window[channel].to_numpy(),
                    "beta": butter_bandpass_filter(
                        spikes_window[channel], 12, 30, fs=1000
                    ),
                    "gamma": butter_bandpass_filter(
                        spikes_window[channel], 30, 80, fs=1000
                    ),
                    "theta": butter_bandpass_filter(
                        spikes_window[channel], 4, 12, fs=1000
                    ),
                }
                this_window[channel] = channel_data

        epoch_data = np.array(
            [this_window[channel]["raw"] for channel in list(lfp.spikes.columns)]
        )
        data.append(epoch_data)
        metadata_list.append({"letter": letter_str, "digit": digit_str})

        data_by_trial_type[trial_key][len(data_by_trial_type[trial_key])] = this_window

    # Make mne epochs for each window based on the shortest window
    min_len = min([d.shape[1] for d in data])
    shortened = []
    for d in data:
        size = d.shape[1]
        d = d[:, size - min_len :]
        shortened.append(d)

    data_array = np.stack(shortened, axis=0)
    epochs = mne.EpochsArray(data_array, info)
    epochs.metadata = pd.DataFrame(metadata_list)

    # Now, you can easily filter epochs based on metadata
    ch1 = "LFP2_vHp"
    ch2 = "LFP4_AON"
    epochs_b1 = epochs['letter == "b" and digit == "1"']
    index_lfp1 = epochs_b1.ch_names.index(ch1)
    index_lfp3 = epochs_b1.ch_names.index(ch2)
    data_b1 = epochs_b1.get_data()
    two_channel_data = data_b1[:, [index_lfp1, index_lfp3], :]
    freqs = np.arange(1, 100, 1)
    n_cycles = freqs / 2.0

    con = mne_connectivity.spectral_connectivity_epochs(
        two_channel_data,
        method="coh",
        sfreq=int(lfp.fs),
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=n_cycles,
        verbose=True,
        indices=(np.array([0]), np.array([1])),
    )

    coh = con.get_data() # The coherence matrix, shape will be (n_freqs, n_times)
    times = epochs_b1.times
    plt.imshow(
        np.squeeze(coh),
        extent=(times[0], times[-1], freqs[0], freqs[-1]),
        aspect="auto",
        origin="lower",
        cmap="jet",
    )
    plt.title(f"{ch1} vs {ch2} \n"
              f"Black, Correct", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    # colorbar units are "coherence"
    plt.colorbar(label="Coherence")
    plt.savefig(save_path / "coh.png", dpi=300, bbox_inches="tight", pad_inches=0.1, transparent=True)

    x = 2
