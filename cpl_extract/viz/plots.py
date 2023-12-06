from __future__ import annotations

import os

import mne
from mne import Epochs
from mne_connectivity import spectral_connectivity_epochs

import matplotlib
from matplotlib import pyplot as plt, gridspec
import numpy as np
from pathlib import Path

from scipy.interpolate import interp1d
from scipy.stats import ttest_rel

from cpl_extract.logger import logger


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
    cax = ax.imshow(
        coh,
        extent=(times[0], times[-1], freqs[0], freqs[-1]),
        aspect="auto",
        origin="lower",
        cmap="jet",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Coherence Value", rotation=270, labelpad=15)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(False)
    plt.tight_layout()
    save_figure(session_path / filename, overwrite=True, fail_silently=True)


def plot_coh(epoch_object: Epochs, freqs=(5, 100)):
    # Remaining code for spectral_connectivity_epochs and plotting
    indices = (np.array([0]), np.array([1]))
    freqs = np.arange(freqs[0], freqs[1], 2)
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


def plot_custom_data(data, times, ch_names, start=None, duration=None, fs=None):
    if start is None:
        start = 0
    if duration is None:
        duration = times[-1] - start
    if fs is None:
        fs = 1 / (times[1] - times[0])

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
        ax[i].set_ylabel("Voltage (V)", **bold_font)

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


def plot_all_epoch(epoch, title, savepath=None):
    for epoch_idx, (epoch_signal_ch1, epoch_signal_ch2) in enumerate(epoch):
        window_size = 100
        # coh_values = rolling_coherence(epoch_signal_ch1, epoch_signal_ch2, window_size)
        coh_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        epoch_times = epoch.times
        coh_times = epoch_times[: len(coh_values)]

        interp_func = interp1d(coh_times, coh_values, kind="cubic")
        new_coh_times = np.linspace(coh_times[0], coh_times[-1], 500)
        new_coh_values = interp_func(new_coh_times)

        # Create grid and subplots
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 0.25, 3])
        gs.update(hspace=0.0)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        # General settings
        ax1.set_title(f"{title}, trial {epoch_idx}")
        for ax in [ax1, ax2]:
            ax.grid(False)

        # Plotting data
        colormap = matplotlib.colormaps["jet"]
        norm = plt.Normalize(0, 1.5)

        # Coherence as fill_between
        num_segments = len(new_coh_times) - 1
        for seg in range(num_segments):
            color_value = norm(new_coh_values[seg])
            ax1.fill_between(
                new_coh_times[seg : seg + 2],
                0,
                new_coh_values[seg : seg + 2],
                color=colormap(color_value),
            )

        # Coherence as heatmap
        ax2.imshow(
            [new_coh_values],
            aspect="auto",
            cmap=colormap,
            norm=norm,
            extent=[new_coh_times[0], new_coh_times[-1], 0, 1],
        )

        # Signal plot
        ax3.plot(
            epoch.times,
            epoch_signal_ch1 * 1000,
            linestyle="-",
            linewidth=1,
            color="black",
        )
        ax3.plot(
            epoch.times,
            epoch_signal_ch2 * 1000,
            linestyle="--",
            linewidth=1,
            color="r",
        )

        xlim = [new_coh_times[0], new_coh_times[-1]]
        ax1.set_xlim(xlim)
        ax1.set_ylim([0, 1.5])
        ax2.set_xlim(xlim)
        ax3.set_xlim([-1, 0])

        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])

        xticks = np.linspace(-1, 0, 11)
        xticklabels = np.linspace(-1000, 0, 11).astype(int).tolist()
        xticklabels[-1] = "DIG"

        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xticklabels)

        ax3.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        if savepath:
            plt.savefig(f"{savepath}/{epoch_idx}.png")
            logger.info(f"Saved {savepath}/{epoch_idx}.png")
            plt.close("all")
        else:
            plt.show()
