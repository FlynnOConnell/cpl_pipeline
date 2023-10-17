# %%
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.fftpack import fft
from scipy.integrate import simps
from scipy.signal import coherence, welch

from spk2extract.logs import logger
from spk2extract.spk_io import spk_h5
from spk2extract.extraction.cluster import filter_signal
import mne
import mne_connectivity

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


class LfpSignal:
    def __init__(
        self,
        spikes_df,
        times_df,
        fs,
        bandpass=None,
        event_arr=None,
        ev_times_arr=None,
        filename=None,
        exclude=(),
    ):
        self._exclude = exclude
        self.spikes: pd.DataFrame = spikes_df
        for col in exclude:
            if col in self.spikes.columns:
                self.spikes.drop(col, axis=1, inplace=True)
        self.times: pd.DataFrame = times_df
        self.events: np.ndarray | None = event_arr
        self.event_times: np.ndarray | None = ev_times_arr
        self.fs: float = fs
        self._filtered = False
        self._bandpass: tuple = bandpass
        self._bands: dict = {
            "Delta": (1, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta": (12, 30),
            "Gamma": (30, 100),
        }
        self.filename: str | Path | None = filename
        self.analysis_results: dict = {}
        self.fft_data: pd.DataFrame | None = None
        self.welch_data: pd.DataFrame | None = None
        self.calculate_fft_data()
        self.calculate_welch_data()

    @property
    def spikes(self):
        filtered_spikes = self._spikes.copy()
        for col in self.exclude:
            if col in filtered_spikes.columns:
                filtered_spikes.drop(col, axis=1, inplace=True)
        if self._filtered:
            for col in filtered_spikes.columns:
                filtered_spikes[col] = filter_signal(
                    filtered_spikes[col], self.fs, self.bandpass
                )
        return filtered_spikes

    @spikes.setter
    def spikes(self, spikes_df):
        self._spikes = spikes_df
        self._filtered = False

    @property
    def exclude(self):
        return self._exclude

    @exclude.setter
    def exclude(self, exclude_list):
        self._exclude = exclude_list

    def wrapper_welch(self, col):
        freq, Pxx = welch(col, fs=self.fs)
        return pd.Series(Pxx, index=freq)

    def calculate_welch_data(self):
        self.welch_data = self.spikes.apply(self.wrapper_welch)

    def __repr__(self):
        return (
            f"LfpSignal: {self.filename if self.filename else 'No filename provided'}"
        )

    def __str__(self):
        return (
            f"LfpSignal: {self.filename if self.filename else 'No filename provided'}"
        )

    @property
    def filtered(self):
        return self._filtered

    @property
    def bandpass(self):
        return self._bandpass

    @bandpass.setter
    def bandpass(self, bandpass: tuple):
        assert len(bandpass) == 2, "Bandpass must be a tuple of length 2"
        self._bandpass = bandpass
        self.filter()

    @property
    def bands(self):
        return self._bands

    @bands.setter
    def bands(self, new_bands: dict):
        self._bands = new_bands

    def filter(self):
        logger.info(f"Filtering data with bandpass {self.bandpass}")
        for col in self.spikes.columns:
            self.spikes[col] = filter_signal(self.spikes[col], self.fs, self.bandpass)
        self._filtered = True

    def get_windows(
        self, pre_time: float = 0.5, post_time: float = 0.5
    ) -> Generator[str, str, tuple] | Generator[None]:
        """
        Returns a generator that yields a tuple of (letter, digit, time_window) for each event pair.

        Returns
        -------
        Generator[str, str, tuple] | Generator[None]
            A generator that yields a tuple of (letter, digit, time_window) for each event pair.

        """
        if not ensure_alternating(self.events):
            yield None, "Events array does not alternate between letters and digits."

        for i in range(0, len(self.events) - 1, 2):
            letter = self.events[i]
            digit = self.events[i + 1]

            time_digit = self.event_times[i + 1] * self.fs
            pre_time_idx = int(time_digit - (pre_time * self.fs))
            post_time_idx = int(time_digit + (post_time * self.fs))

            if letter in ["b", "w"]:
                spikes_window = self.spikes.loc[pre_time_idx:post_time_idx]
                yield letter, digit, spikes_window

    def get_fft_values(self, signal):
        if hasattr(signal, "values"):
            signal = signal.values  # pd object isn't compatible with fft
        N = len(signal)
        fft_values = fft(signal)
        frequencies = np.fft.fftfreq(N, 1 / self.fs)
        return frequencies[: N // 2], 2.0 / N * np.abs(fft_values[: N // 2])

    def get_coherence(self, signal1, signal2):
        f, Cxy = coherence(signal1, signal2, fs=self.fs)
        return f, Cxy

    def get_band_power(self, signal, band):
        if hasattr(signal, "values"):
            signal = signal.values  # Convert to NumPy array if it's a Pandas Series
        freqs, fft_vals = self.get_fft_values(signal)
        band_freqs = [freq for freq in freqs if band[0] <= freq <= band[1]]
        band_fft_vals = [
            fft_val
            for fft_val, freq in zip(fft_vals, freqs)
            if band[0] <= freq <= band[1]
        ]
        return simps(band_fft_vals, band_freqs)

    def wrapper(self, col):
        freq, fft_val = self.get_fft_values(col)
        return pd.Series(fft_val, index=freq)

    def calculate_fft_data(self):
        self.fft_data = self.spikes.apply(self.wrapper)


if __name__ == "__main__":
    data_path = Path().home() / "data" / "extracted"
    file = list(data_path.glob("*0609*.h5"))[0]
    df_s, df_t, events, event_times = get_data(file)
    unique_events = np.unique(events)
    key = {
        "0": "dug incorrectly",
        "1": "dug correctly",
        "x": "crossed over (no context)",
        "b": "crossed over into the black room",
        "w": "crossed over into the white room",
    }

    lfp = LfpSignal(
        df_s, df_t, event_arr=events, ev_times_arr=event_times, fs=2000, filename=file
    )
    lfp.bandpass = (0.1, 100)

    event = {}
    for letter, digit, window in lfp.get_windows():
        df = window
        df2 = df.loc[:, :]

    temp = 1
    # powerbands = {
    #     "Delta": (1, 4),
    #     "Theta": (4, 8),
    #     "Alpha": (8, 12),
    #     "Beta": (12, 30),
    #     "Gamma": (30, 100),
    # }
    x = 4
