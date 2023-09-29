.. _wf_sample:

Spike Waveform Samples Explanation
==================================

When processing electrophysiological data to extract spike waveforms, it's crucial to understand how many samples each extracted spike waveform will contain.

Setting Up Snapshot Duration
----------------------------

In the extract_waveforms() code, a "snapshot" around each detected spike is defined using the parameter `spike_snapshot`, set to `(0.2, 0.6)` milliseconds by default. This means that each snapshot will contain:

- 0.2 milliseconds of data before the spike
- 0.6 milliseconds of data after the spike
- The spike itself

Calculating Number of Samples
-----------------------------

Given a sampling rate of 18,518 Hz, the duration of each sample would be :math:`\frac{1}{18518}` seconds or approximately 0.054 milliseconds.

1. Pre-spike samples: :math:`\frac{0.2}{0.054} \approx 3.7`, rounded to 4 samples
2. Post-spike samples: :math:`\frac{0.6}{0.054} \approx 11.1`, rounded to 11 samples
3. Adding 1 for the spike itself gives you :math:`4 + 11 + 1 = 16` samples
