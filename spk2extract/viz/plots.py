import matplotlib
from mne import Epochs
from mne_connectivity import spectral_connectivity_epochs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def plot_coh(epoch_object: Epochs, freqs=None):
    # Connectivity Analysis
    indices = (np.array([0]), np.array([1]))  # Channel indices for connectivity
    freqs = np.arange(5, 100) if freqs is None else freqs
    n_cycles = freqs / 2

    con = spectral_connectivity_epochs(
        epoch_object,
        indices=indices,
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
    title = f"Coherence {epoch_object.ch_names[0]} vs {epoch_object.ch_names[1]}"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def plot_custom_data(data, times, ch_names, start, duration, fs=None):

    start_idx = int(start * fs)
    end_idx = int((start + duration) * fs)
    times = times[start_idx:end_idx] * 1000  # Convert to milliseconds
    times = times - times[0]
    data = data[:, start_idx:end_idx]

    fig, ax = plt.subplots(len(ch_names), 1, sharex=True, figsize=(15, len(ch_names) * 2))
    bold_font = {"fontweight": "bold"}

    for i, ch_name in enumerate(ch_names):
        ax[i].plot(times, data[i, :], color="black")
        ax[i].set_title(ch_name, **bold_font)
        ax[i].axhline(0, color="gray", linewidth=0.8)
        ax[i].set_facecolor("none")
        ax[i].grid(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].set_ylabel("Voltage (mV)", **bold_font)

    ax[-1].set_xlabel("Time (ms)", **bold_font)
    plt.tight_layout()
    plt.show()


def plot_processing_steps(raw_list, titles, start, duration, scalings, channel_name):
    n = len(raw_list)

    fig, ax = plt.subplots(n, 1, figsize=(15, 4 * n), sharex=True)

    bold_font = {"fontweight": "bold"}

    for i, (raw, title) in enumerate(zip(raw_list, titles)):
        ch_idx = raw.ch_names.index(channel_name)
        data, times = raw[
            ch_idx,
            int(start * raw.info["sfreq"]) : int(
                (start + duration) * raw.info["sfreq"]
            ),
        ]
        data = data.flatten()
        data *= scalings

        # convert times to ms
        times = times - times[0]
        times = times * 1000
        ax[i].plot(times + start, data, color="black")
        ax[i].set_title(f"{channel_name} - {title}", **bold_font)
        ax[i].set_ylabel("Voltage (mV)", **bold_font)
        ax[i].axhline(0, color="gray", linewidth=0.8)
        ax[i].set_facecolor("none")
        ax[i].grid(False)
        #     prevent the border around the axis from being drawn
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

    ax[-1].set_xlabel("Time (ms)", **bold_font)
    plt.tight_layout()
    plt.show()

