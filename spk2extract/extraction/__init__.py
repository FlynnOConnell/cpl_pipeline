from .spike2 import Spike2Data, WaveData

__all__ = ["Spike2Data", "WaveData"]
#       fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
#         fig.suptitle(
#             f"Beta Filtered LFP: {letter_map.get(letter_str)}, {digit_map.get(digit_str)}",
#             fontsize=20,
#             fontweight="bold",
#             y=0.95,
#             x=0.5,
#             ha="center",
#             va="top",
#         )
# lfp = LfpSignal(
#     resampled_lfp_df,
#     1000,
#     event_arr=ev,
#     ev_times_arr=event_times,
#     filename=file,
#     exclude=["LFP1_AON", "LFP2_AON"],
# )
# lfp.filter(15, 100)
# lfp.notch()
# windowed_data = {}
#
# letter_map = {"b": "Black", "w": "White"}
# digit_map = {"0": "Incorrect dig", "1": "Correct dig"}
#
# x = 0
# for letter_str, digit_str, spikes_window in lfp.get_windows(0.5, 0.5, "all"):
#     x += 1
#     this_window = {}
#
#     fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
#     fig.suptitle(
#         f"Beta Filtered LFP: {letter_map.get(letter_str)}, {digit_map.get(digit_str)}",
#         fontsize=20,
#         fontweight="bold",
#         y=0.95,
#         x=0.5,
#         ha="center",
#         va="top",
#     )
#
#     channel_pairs = [("LFP1_vHp", "LFP3_AON"), ("LFP2_vHp", "LFP4_AON")]
#
#     ax_idx = 0
#     for ch1, ch2 in channel_pairs:
#         beta_filtered_list = []
#         for channel in [ch1, ch2]:
#             channel_data = {"raw": spikes_window[channel]}
#
#             idx_min, idx_max = spikes_window.index.min(), spikes_window.index.max()
#             xticks = np.linspace(idx_min, idx_max, 11)
#             xticklabels = np.linspace(0, (idx_max - idx_min) / 2000, 11).round(2)
#
#             ymin, ymax = spikes_window[channel].min(), spikes_window[channel].max()
#
#             beta_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 12, 30, fs=1000
#             )
#             gamma_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 30, 80, fs=1000
#             )
#             theta_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 4, 12, fs=1000
#             )
#             channel_data["beta"] = beta_filtered
#             channel_data["gamma"] = gamma_filtered
#             channel_data["theta"] = theta_filtered

# ax[ax_idx].plot(
#     spikes_window.index,
#     spikes_window[channel],
#     label="Raw",
#     color="gray",
#     linewidth=1,
#     alpha=0.5,
#     zorder=1,
# )
# ax[ax_idx].plot(
#     spikes_window.index,
#     beta_filtered,
#     label="Beta",
#     color="black",
#     linewidth=2,
#     zorder=2,
# )
#
# ax[ax_idx].set_title(f"{channel}")
# ax[ax_idx].set_xticks(xticks)
# ax[ax_idx].set_xticklabels(xticklabels)
# ax[ax_idx].set_ylabel("Amplitude (V)")
# ax[ax_idx].set_xlim(idx_min, idx_max)
# ax[ax_idx].set_ylim(ymin, ymax)
# ax[ax_idx].legend()
# ax_idx += 1
#
# lfp = LfpSignal(
#     resampled_lfp_df,
#     1000,
#     event_arr=ev,
#     ev_times_arr=event_times,
#     filename=file,
#     exclude=["LFP1_AON", "LFP2_AON"],
# )
# lfp.filter(15, 100)
# lfp.notch()
# windowed_data = {}
#
# letter_map = {"b": "Black", "w": "White"}
# digit_map = {"0": "Incorrect dig", "1": "Correct dig"}
#
# x = 0
# for letter_str, digit_str, spikes_window in lfp.get_windows(0.5, 0.5, "all"):
#     x += 1
#     this_window = {}
#
#     fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
#     fig.suptitle(
#         f"Beta Filtered LFP: {letter_map.get(letter_str)}, {digit_map.get(digit_str)}",
#         fontsize=20,
#         fontweight="bold",
#         y=0.95,
#         x=0.5,
#         ha="center",
#         va="top",
#     )
#
#     channel_pairs = [("LFP1_vHp", "LFP3_AON"), ("LFP2_vHp", "LFP4_AON")]
#
#     ax_idx = 0
#     for ch1, ch2 in channel_pairs:
#         beta_filtered_list = []
#         for channel in [ch1, ch2]:
#             channel_data = {"raw": spikes_window[channel]}
#
#             idx_min, idx_max = spikes_window.index.min(), spikes_window.index.max()
#             xticks = np.linspace(idx_min, idx_max, 11)
#             xticklabels = np.linspace(0, (idx_max - idx_min) / 2000, 11).round(2)
#
#             ymin, ymax = spikes_window[channel].min(), spikes_window[channel].max()
#
#             beta_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 12, 30, fs=1000
#             )
#             gamma_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 30, 80, fs=1000
#             )
#             theta_filtered = butter_bandpass_filter(
#                 spikes_window[channel], 4, 12, fs=1000
#             )
#             channel_data["beta"] = beta_filtered
#             channel_data["gamma"] = gamma_filtered
#             channel_data["theta"] = theta_filtered
