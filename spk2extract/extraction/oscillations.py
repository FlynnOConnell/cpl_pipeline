import numpy as np
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
from spk2extract.spk_io import spk_h5
from matplotlib import pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    print("Filter coefficients b:", b)
    print("Filter coefficients a:", a)
    # Break down into two steps: first high-pass, then low-pass
    b, a = butter(order, lowcut / (0.5 * fs), btype='high')
    data = filtfilt(b, a, data)
    b, a = butter(order, highcut / (0.5 * fs), btype='low')
    y = filtfilt(b, a, data)
    return y

path = Path().home() / "spk2extract" / "h5"
files = list(path.glob("*.h5"))
for file in files:
    h5_file = spk_h5.read_h5(path / file)
    events = h5_file["event"]["Event"]
    lfp1 = h5_file["data"]["LFP1_OB"]["LFP1_OB"][0]
    lfp1_ts = h5_file["data"]["LFP1_OB"]["LFP1_OB"][1]

    fs = h5_file["metadata_channel"]["LFP1_OB"]["fs"]
    evt = events[0]

    lowcut = 1.0  # 1 Hz
    highcut = 500.0  # 500 Hz
    lfp1_filtered = butter_bandpass_filter(lfp1, lowcut, highcut, fs, order=4)
    evt_idx = int(evt * fs)
    window_size = int(0.1 * fs)  # 100ms window, converted to array indices
    time_segment = [evt_idx - window_size, evt_idx + window_size]

    time_axis = np.linspace(-window_size, window_size, 2 * window_size) / fs * 1000

    # Plot lfp with title, x and y label using plt.subplots()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, lfp1_filtered[time_segment[0] : time_segment[1]])
    ax.set(xlabel="Time post odor onset (ms)", ylabel="Voltage (V)", title=f"{file.name} - LFP1 - {evt} s")
    plt.show()