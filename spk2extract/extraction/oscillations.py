from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.signal import butter, filtfilt, coherence
from scipy.fftpack import fft
import seaborn as sns

import spk2extract
from spk2extract.spk_io import spk_h5

logger = spk2extract.logger
logger.setLevel("INFO")

sns.set_style("darkgrid")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Break down into two steps: first high-pass, then low-pass
    b, a = butter(order, lowcut / (0.5 * fs), btype="high")
    data = filtfilt(b, a, data)
    b, a = butter(order, highcut / (0.5 * fs), btype="low")
    y = filtfilt(b, a, data)
    return y


def get_data(path: Path | str):
    path = Path(path)
    files = list(path.glob("*dk3*"))
    file = files[0]
    spikes_df = pd.DataFrame()
    times_df = pd.DataFrame()
    h5_file = spk_h5.read_h5(path / file)
    spikedata = h5_file["spikedata"]
    for chan, data in spikedata.items():
        if chan in ["VERSION", "CLASS", "TITLE"]:
            continue
        logger.info(f"Channel: {chan}")
        spikes = data["spikes"]["spikes"]
        times = data["times"]["times"]
        spikes_df[chan] = spikes
        times_df[chan] = times
    return spikes_df, times_df


class LfpSignal:
    def __init__(self, spikes_df, times_df, fs):
        self.spikes: pd.DataFrame = spikes_df
        self.times: pd.DataFrame = times_df
        self.fs: float = fs
        self._bandpass: tuple = ()
        self.filtered = False
        self.analysis_results = {}  # To hold analysis results for each channel

    @property
    def bandpass(self):
        return self._bandpass

    @bandpass.setter
    def bandpass(self, bandpass: tuple):
        assert len(bandpass) == 2, "Bandpass must be a tuple of length 2"
        self._bandpass = bandpass

    def filter(self):
        if not self._bandpass:
            logger.debug("Bandpass not set, skipping filtering")
            return
        low, high = self._bandpass
        self.spikes = self.spikes.apply(
            lambda col: butter_bandpass_filter(col, low, high, self.fs, order=4)
        )
        self.filtered = True

    def get_fft_values(self, signal):
        if hasattr(signal, "values"):
            signal = signal.values  # pd object isn't compatible with fft
        # signal = signal - np.mean(signal)  # remove DC component
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

    def calculate_band_powers(self, bands):
        for channel in self.spikes.columns:
            self.analysis_results[channel] = {}
            for band_name, freq_range in bands.items():
                self.analysis_results[channel][band_name] = self.get_band_power(
                    self.spikes[channel], freq_range
                )

    def plot_coherence_for_pairs(self):
        channel_pairs = itertools.combinations(self.spikes.columns, 2)
        for chan1, chan2 in channel_pairs:
            f, Cxy = self.get_coherence(self.spikes[chan1], self.spikes[chan2])
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(f, Cxy)
            ax.set_title(f"Coherence between {chan1} and {chan2}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Coherence")
            ax.grid(True)
        plt.show()


if __name__ == "__main__":
    data_path = Path().home() / "spk2extract" / "h5"
    df_s, df_t = get_data(data_path)
    lfp = LfpSignal(df_s, df_t, 2000)
    lfp.bandpass = (0.1, 500)
    lfp.filter()
    lfp.plot_coherence_for_pairs()

    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 100),
    }
    # This will populate the analysis_results attribute
    lfp.calculate_band_powers(bands)
    logger.info(lfp.analysis_results)
    x = 4
