from __future__ import annotations

from itertools import combinations

import mne_connectivity

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mne.time_frequency import psd_array_welch
from mne_connectivity import spectral_connectivity_epochs
from scipy.stats import zscore, sem
from sklearn.manifold import spectral_embedding  # noqa

from spk2extract.analysis.stats import extract_file_info
from spk2extract.logs import logger
from spk2extract.spk_io.utils import read_npz_as_dict
from spk2extract.utils import extract_common_key
from spk2extract.viz import helpers
from spk2extract.viz import plots

sns.set_style("darkgrid")

logger.setLevel("INFO")


def _validate_events(event_arr, event_id):
    """
    Removes events that occur less than 3 times.

    Parameters:
    -----------
    event_arr (np.array): Array of events
    event_id (dict): Dictionary of event ids

    Returns:
    --------
    valid_times (np.array): Array of valid events
    valid_event_id (dict): Dictionary of valid event ids
    """
    assert event_arr.shape[1] == 3, "Event array should have 3 columns"

    valid_event_id = {}
    for event_name, event_val in event_id.items():
        occurrences = sum(event_arr[:, -1] == event_val)
        if occurrences > 2:
            valid_event_id[event_name] = event_val

    # Remove invalid events from times array
    valid_values = list(valid_event_id.values())
    valid_times = np.array([row for row in event_arr if row[-1] in valid_values])

    return valid_times, valid_event_id

def get_baseline(resp_signal, resp_times, wave_signal, first):
    wave_signal = wave_signal.get_data()
    first_event_time = first
    baseline_size = 10

    respiratory_signal = np.array(resp_signal[:int(first_event_time)], dtype=float)  # 1 Hz
    respiratory_times = np.array(resp_times[:int(first_event_time)], dtype=float)  # 1 Hz

    wave_signal = np.array(wave_signal[:, :int(first_event_time * 2000)], dtype=float)

    # Step 1: Identify slow breathing segment using rolling variance
    rolling_variance = pd.Series(respiratory_signal).rolling(window=baseline_size).var()
    rolling_variance = rolling_variance.dropna()
    min_var_index = rolling_variance.idxmin()
    slowest_breathing_time = respiratory_times[min_var_index]

    # Convert this time to an index in spikes_arr
    slowest_breathing_index_spikes_arr = int(
        slowest_breathing_time * 2000)  # Conversion factor due to different sampling rates

    # Step 2: Extract corresponding segment from spikes_arr
    segment_data = wave_signal[:,
                   slowest_breathing_index_spikes_arr:slowest_breathing_index_spikes_arr + baseline_size * 2000]

    return np.array(segment_data)

def get_master_df():
    data_path = Path().home() / "data" / ".cache"
    animals = list(data_path.iterdir())

    master = pd.DataFrame()
    for animal_path in animals:

        cache_animal_path = data_path / animal_path.name
        cache_animal_path.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(list(animal_path.glob("*_raw*.fif")), key=extract_common_key)
        event_files = sorted(list(animal_path.glob("*_eve*.fif")), key=extract_common_key)
        event_id = sorted(list(animal_path.glob("*_id_*.npz")), key=extract_common_key)
        norm_signal = sorted(animal_path.glob("*_respiratory_signal*.npy"), key=extract_common_key)
        norm_times = sorted(animal_path.glob("*_respiratory_times*.npy"), key=extract_common_key)
        assert len(raw_files) == len(event_files) == len(event_id)

        metadata = [extract_file_info(raw_file.name) for raw_file in raw_files]
        raw_data = [mne.io.read_raw_fif(raw_file) for raw_file in raw_files]
        event_data = [mne.read_events(event_file) for event_file in event_files]
        event_id_dicts = [read_npz_as_dict(id_file) for id_file in event_id]

        norm_signal = [np.load(str(signal_file)) for signal_file in norm_signal]
        norm_times = [np.load(str(time_file)) for time_file in norm_times]

        first_event_times = [event[0][0] for event in event_data]
        baseline_segment = [get_baseline(resp_signal, resp_times, raw_arr, first_event) for
                            resp_signal, resp_times, raw_arr, first_event in
                            zip(norm_signal, norm_times, raw_data, first_event_times)]
        # map each baseline channel to its baseline segment

        ev_id_dict = [
            {k: v for k, v in zip(ev_id_dict["keys"], ev_id_dict["values"])}
            for ev_id_dict in event_id_dicts
        ]

        ev_start_holder = []
        ev_end_holder = []
        for (
                arr
        ) in (
                event_data
        ):  # separate the start and end events to work with mne event structures [start, 0, id]
            start_events = np.column_stack(
                [arr[:, 0], np.zeros(arr.shape[0]), arr[:, 2]]
            ).astype(int)
            end_events = np.column_stack(
                [arr[:, 1], np.zeros(arr.shape[0]), arr[:, 2]]
            ).astype(int)

            ev_start_holder.append(start_events)
            ev_end_holder.append(end_events)

        all_data_dict = {  # each v is a list for each file, sorted to be the smae order
            "raw": raw_data,
            "start": ev_start_holder,
            "end": ev_end_holder,
            "event_id": ev_id_dict,
            "baseline": baseline_segment,
        }
        metadata_df = pd.concat(metadata).reset_index(drop=True)
        all_data_df = pd.DataFrame(all_data_dict)
        all_data_df = pd.concat([metadata_df, all_data_df], axis=1)
        master = pd.concat([master, all_data_df], axis=0)
    return master

def preprocess_raw(raw_signal, fmin=1, fmax=100):
    raw: mne.io.RawArray = raw_signal.copy()
    raw.load_data()
    raw.filter(fmin, fmax, fir_design="firwin")
    raw.resample(1000)
    raw.notch_filter(np.arange(60, 161, 60), fir_design="firwin")
    raw.set_eeg_reference(ref_channels=["Ref"])
    raw.drop_channels(["Ref"])
    return raw

def update_df_with_epochs(df, fmin, fmax, tmin, tmax):
    epochs_holder, con_holder = [], []
    for idx, row in df.iterrows():
        epoch, con_data = process_epoch(row['raw'], row['start'], row['event_id'], row['baseline'], fmin, fmax, tmin,
                                        tmax)
        epochs_holder.append(epoch)

        con_holder.append(con_data)
    df['epoch'] = epochs_holder
    df['con_arr'] = con_holder
    return df

def process_epoch(raw_arr, events, event_id, baseline, fmin, fmax, tmin, tmax):
    assert fmin < fmax and tmin < tmax, "Ensure fmin < fmax and tmin < tmax"

    raw = preprocess_raw(raw_arr, fmin, fmax)
    means = np.mean(baseline, axis=1)
    stds = np.std(baseline, axis=1)
    means = means[:-1]
    stds = stds[:-1]

    valid_events, valid_event_id = _validate_events(events, event_id)
    raw = raw.filter(fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1)

    epochs = mne.Epochs(raw, valid_events, valid_event_id, tmin, tmax, baseline=None, event_repeated="drop",
                        preload=True)

    if epochs is None:
        raise ValueError("Epoch is None")

    freqs_arange = np.arange(fmin, fmax)

    con = spectral_connectivity_epochs(
        epochs,
        indices=(np.array([0]), np.array([1])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs_arange,
        cwt_n_cycles=freqs_arange / 2,
        sfreq=epochs.info["sfreq"],
        n_jobs=1,
    )
    return epochs, con

def main(use_parallel=True):
    helpers.update_rcparams()
    data = get_master_df().reset_index(drop=True)
    return data

def optimize_psd_params(sfreq, time_series_length, desired_resolution=None):
    """
    Optimize PSD calculation parameters.

    Parameters:
    - sfreq: Sampling frequency
    - time_series_length: Length of the time series data
    - desired_resolution: Desired frequency resolution (optional)

    Returns:
    - n_fft: Optimal FFT length
    - n_per_seg: Optimal number of samples per segment
    - n_overlap: Optimal number of overlap samples
    """

    # Choose n_fft as the next power of 2 greater than time_series_length for computational efficiency
    n_fft = 2 ** np.ceil(np.log2(time_series_length))
    n_fft = int(min(n_fft, time_series_length))

    if desired_resolution:
        n_fft = int(sfreq / desired_resolution)

    n_per_seg = n_fft  # You can make this smaller to increase the number of segments averaged over
    n_overlap = int(n_per_seg / 2)

    return n_fft, n_per_seg, n_overlap

def compute_coherence_base(epoch_data, idxs, freqs):

    if hasattr(epoch_data, "info"):
        sfreq = epoch_data.info["sfreq"]
    else:
        sfreq = 1000

    epoch_connectivity = spectral_connectivity_epochs(
        epoch_data,
        indices=(np.array([idxs[0]]), np.array([idxs[1]])),
        method="coh",
        mode="cwt_morlet",
        cwt_freqs=freqs,
        cwt_n_cycles=freqs / 2,
        sfreq=sfreq,
        n_jobs=1,
    )
    return epoch_connectivity

def set_event_id(epoch):
    epoch.event_id = all_event_keys
    return epoch

def determine_group(chan_pair_names):
    """
    Determines whether a pair of channels is within or between brain regions.

    Parameters
    ----------
    chan_pair_names : tuple
        Tuple of channel names.

    Returns
    -------
    str
        'within' or 'between'

    """
    if any(all(x in chan_pair_names for x in w) for w in brain_regions_within):
        return 'within'
    elif any(all(x in chan_pair_names for x in b) for b in brain_regions_between):
        return 'between'
    else:
        raise ValueError(f"Invalid pair name: {chan_pair_names}")

if __name__ == "__main__":

    data = main(use_parallel=False)

    epoch_tmin, epoch_tmax = -1, 0
    pre_fmin, pre_fmax = 1, 100

    brain_regions_within = [('LFP1_vHp', 'LFP2_vHp'), ('LFP3_AON', 'LFP4_AON')]
    brain_regions_between = [('LFP1_vHp', 'LFP3_AON'), ('LFP1_vHp', 'LFP4_AON'), ('LFP2_vHp', 'LFP3_AON'),
                             ('LFP2_vHp', 'LFP4_AON')]
    brain_regions_all = ['LFP1_vHp', 'LFP2_vHp', 'LFP3_AON', 'LFP4_AON']

    all_animals = data["Animal"].unique()
    all_event_keys = {"b_1": 1, "b_0": 2, "w_1": 3, "w_0": 4, "x_1": 5, "x_0": 6}
    lfp_bands = {'beta': (12, 30), 'gamma': (30, 80), 'theta': (4, 12)}
    chan_indices = np.array(list(combinations(range(4), 2)))

    all_animals_data = {}
    for animal in all_animals:

        all_animals_data[animal] = {}
        animal_df = data[data["Animal"] == animal]
        for context in ["context", "nocontext"]:

            all_animals_data[animal][context] = {}

            df = update_df_with_epochs(animal_df[animal_df["Context"] == context], pre_fmin, pre_fmax, epoch_tmin,
                                       epoch_tmax)

            baseline = df['baseline'].tolist()  # list, item for each session
            for _bl in baseline:
                _bl = _bl[:-1]

            # todo: bye forbidden lambda
            epochs = list(map(lambda e: set_event_id(e), df['epoch'].tolist()))  # list, item for each session
            baseline = df['baseline'].tolist()  # list, item for each session

            concatenated_epochs = mne.concatenate_epochs(epochs)
            concatenated_bl = np.concatenate(baseline, axis=1)
            concatenated_bl = concatenated_bl.reshape(1, concatenated_bl.shape[0], concatenated_bl.shape[1])
            cells = concatenated_epochs.ch_names

            # elim any dropped channels
            channel_indices = [cells.index(ch_name) for ch_name in concatenated_epochs.ch_names]
            reshaped_bl = concatenated_bl[:, channel_indices, :]

            # bundle baseline into epochs
            final_bl = mne.EpochsArray(reshaped_bl, concatenated_epochs.info)
            for ch_pair in combinations(range(len(brain_regions_all)), 2):

                pair_name = (brain_regions_all[ch_pair[0]], brain_regions_all[ch_pair[1]])
                group = determine_group(pair_name)
                all_animals_data[animal][context][group] = {}

                con = compute_coherence_base(concatenated_epochs, ch_pair, np.arange(pre_fmin, pre_fmax))
                coh = np.squeeze(con.get_data())
                connectivity_freqs = np.array(con.freqs)
                time = np.array(con.times)

                con_bl = compute_coherence_base(final_bl, ch_pair, np.arange(pre_fmin, pre_fmax))
                coh_bl = np.squeeze(con_bl.get_data())
                connectivity_freqs_bl = np.array(con_bl.freqs)
                time_bl = np.array(con_bl.times)

                mean_coh = None
                sem_coh = None
                freqs_coh = None

                for domain in ['time_domain', 'freq_domain']:

                    all_animals_data[animal][context][group][domain] = {}
                    for band, (fmin, fmax) in lfp_bands.items():

                        idx = np.where((connectivity_freqs >= fmin) & (connectivity_freqs <= fmax))[0]
                        idx_bl = np.where((connectivity_freqs_bl >= fmin) & (connectivity_freqs_bl <= fmax))[0]

                        if domain == 'time_domain':
                            mean_coh = np.mean(coh[idx, :], axis=0)
                            mean_coh_bl = np.mean(coh_bl[idx_bl, :], axis=0)

                            sem_coh = np.std(coh[idx, :], axis=0) / np.sqrt(coh[idx, :].shape[0])
                            sem_coh_bl = np.std(coh_bl[idx_bl, :], axis=0) / np.sqrt(coh_bl[idx_bl, :].shape[0])

                            mean_coh = ((mean_coh - mean_coh_bl) / mean_coh_bl) * 100
                            sem_coh = ((sem_coh - sem_coh_bl) / sem_coh_bl) * 100

                        elif domain == 'freq_domain':
                            mean_coh = np.mean(coh[idx, :], axis=1)
                            mean_coh_bl = np.mean(coh_bl[idx_bl, :], axis=1)
                            sem_coh = np.std(coh[idx, :], axis=1) / np.sqrt(coh[idx, :].shape[1])
                            sem_coh_bl = np.std(coh_bl[idx_bl, :], axis=1) / np.sqrt(coh_bl[idx_bl, :].shape[1])

                            mean_coh = ((mean_coh - mean_coh_bl) / mean_coh_bl) * 100
                            sem_coh = ((sem_coh - sem_coh_bl) / sem_coh_bl) * 100

                        freqs_coh = connectivity_freqs[idx]
                        all_animals_data[animal][context][group][domain][band] = {'mean': mean_coh, 'sem': sem_coh,
                                                                                  'freqs': freqs_coh,
                                                                                  'baseline': df['baseline'].tolist()}
                    all_animals_data[animal][context][group]['all'] = {'mean': coh, 'sem': None,
                                                                       'freqs': connectivity_freqs}

        # plotting
        band_color_map = {'beta': 'r', 'gamma': 'b', 'theta': 'g'}
        plots_path = Path().home() / 'data' / 'plots' / 'coh' / 'normalized'
        plots_path.mkdir(parents=True, exist_ok=True)

        for context in ["context", "nocontext"]:
            plots_path_context = plots_path / context
            plots_path_context.mkdir(parents=True, exist_ok=True)
            for group in ['between', 'within']:
                plots_path_group = plots_path_context / group
                plots_path_group.mkdir(parents=True, exist_ok=True)
                fig, (ax_time, ax_freq, ax_all) = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
                ax_time.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                ax_freq.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                ax_time.set_yticks(np.arange(0, 1.1, 0.1))
                ax_freq.set_yticks(np.arange(0, 1.1, 0.1))
                sns.set_style("darkgrid")

                # for domain, ax in [('time_domain', ax_time), ('freq_domain', ax_freq)]:
                for domain, ax in [('time_domain', ax_time)]:
                    for band_name, coh_data in all_animals_data[animal][context][group][domain].items():
                        y_mean = coh_data['mean']
                        y_sem = coh_data['sem']

                        xlim = None
                        if domain == 'time_domain' or band_name == 'all':
                            x = con.times

                        else:
                            x = coh_data['freqs']

                        xlim = (x[0], x[-1])
                        ax.set_xlim(xlim)

                        if band_name != 'all':
                            color = band_color_map[band_name]
                            ax.title(f'{band_name}: {domain}, {context}, {group}', fontsize=14, fontweight='bold')

                            # Plot the mean and SEM shading
                            ax.fill_between(x, (y_mean - y_sem) * 100, (y_mean + y_sem) * 100, alpha=0.2, color=color)
                            ax.plot(x, y_mean * 100, linewidth=2, label=f'{band_name}', color=color)

                        if band_name == 'all':
                            ax.set_ylabel('frequency (hz)')
                            ax.set_yticks(np.arange(0, 100, 10))

                y_mean_all = all_animals_data[animal][context][group]['time_domain']['all']['mean']
                x_all = con.times
                ax_all.imshow(y_mean_all, cmap='jet', aspect='auto', origin='lower', extent=[x_all[0], x_all[-1], 0, 1])
                ax_all.set_xlabel('Time (s)')
                ax_all.set_title(f"", fontsize=14, fontweight='bold')
                ax_all.grid(False)

                ax_time.legend()
                ax_freq.legend()
                plt.show()
                # plt.savefig(plots_path_group / f'{animal}_{context}_{group}.png', dpi=300, bbox_inches='tight',
                #             pad_inches=0.1, )
