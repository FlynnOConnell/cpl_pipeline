import matplotlib
from mne import Epochs
from mne_connectivity import spectral_connectivity_epochs
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

def save_figure(path, overwrite=False, fail_silently=False):
    path = Path(path)
    if path.exists() and not overwrite:
        if fail_silently:
            return
        else:
            raise FileExistsError(f"File {path} already exists.")
    plt.savefig(path)

def plot_2D_coherence(coh, times, freqs, title, filename, session_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(coh, extent=(times[0], times[-1], freqs[0], freqs[-1]), aspect="auto", origin="lower", cmap="jet", interpolation="bilinear")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Coherence Value", rotation=270, labelpad=15)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    save_figure(session_path / filename, overwrite=True, fail_silently=True)
    plt.show()

def plot_3D_coherence(coh, times, freqs):
    X, Y = np.meshgrid(times, freqs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, coh)
    plt.show()

def plot_coh(epoch_object: Epochs, freqs=None):
    # Remaining code for spectral_connectivity_epochs and plotting
    indices = (np.array([0]), np.array([1]))
    freqs = np.arange(5, 100) if freqs is None else np.arange(freqs[0], freqs[1])
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
    coh = con.get_data()  # the connectivity matrix, shape will be (n_freqs, n_times)

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

    fig, ax = plt.subplots(
        len(ch_names), 1, sharex=True, figsize=(15, len(ch_names) * 2)
    )
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
