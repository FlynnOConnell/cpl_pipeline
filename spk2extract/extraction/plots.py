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


def plot_coh(epoch_object: Epochs):
    # Connectivity Analysis
    indices = (np.array([0]), np.array([1]))  # Channel indices for connectivity
    freqs = np.arange(5, 100)
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
    title = f"{epoch_object.ch_names[0]} vs {epoch_object.ch_names[1]}"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def plot_custom_mne_raw(raw, title, start, duration, scalings):
    data, times = raw[
        :,
        int(start * raw.info["sfreq"]) : int((start + duration) * raw.info["sfreq"]),
    ]
    times = times - times[0]
    data *= scalings
    fig, ax = plt.subplots(
        len(raw.ch_names), 1, sharex=True, figsize=(10, len(raw.ch_names) * 1.5)
    )

    bold_font = {"fontweight": "bold"}

    for i, ch_name in enumerate(raw.ch_names):
        ax[i].plot(times + start, data[i, :], color="black")
        ax[i].set_title(ch_name, **bold_font)
        ax[i].set_ylabel("mV", **bold_font)
        ax[i].axhline(0, color="gray", linewidth=0.8)
        ax[i].set_facecolor("none")
        ax[i].grid(False)

    ax[-1].set_xlabel("Time (s)", **bold_font)
    fig.suptitle(title, **bold_font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_processing_steps(raw, title, start, duration, scalings, channel_name):
    ch_idx = raw.ch_names.index(channel_name)
    data, times = raw[
        ch_idx,
        int(start * raw.info["sfreq"]):int((start + duration) * raw.info["sfreq"])
    ]
    data *= scalings

    fig, ax = plt.subplots(figsize=(15, 4))

    bold_font = {"fontweight": "bold"}

    ax.plot(times + start, data, color="black")
    ax.set_title(f"{channel_name} - {title}", **bold_font)
    ax.set_ylabel("Voltage (V)", **bold_font)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_facecolor("none")
    ax.grid(False)

    ax.set_xlabel("Time (s)", **bold_font)
    plt.tight_layout()
    plt.show()
