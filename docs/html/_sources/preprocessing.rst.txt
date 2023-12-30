Analyzing ADC Signals
=====================

The SonPy C++ bindings give us access to some helpful information:

- **Recording Length**: The length of the recording in seconds.
- **Time Base**: The time base of the recording in seconds.
- **Clock Ticks**: The number of clock ticks in the recording.
- **Sample Interval in Clock Ticks**: The number of clock ticks between each sample.

Using this information, we can read the waveform array and gather some insights:

Given Parameters
----------------

- **Recording Length**: \(1080\) seconds
- **Time Base**: :math:`3 \times 10^{-6}` seconds
- **Clock Ticks**: :math:`363172930`
- **Sample Interval in Clock Ticks**: :math:`18`
- **Sample Frequency**: :math:`\frac{1}{18 \times 3 \times 10^{-6}} = 18518.52` Hz
- **Array Size**: :math:`20176274`

Insights
--------

1. **Temporal Resolution**

    * Each point in the array represents :math:`18 \times 3 \times 10^{-6} = 54 \times 10^{-6}` seconds or \(54 \mu s\).
    * For a full action potential lasting about \(2-5 ms\), you would have around :math:`37` to :math:`93` points.
    * In one second, you would have approximately :math:`\frac{1}{54 \times 10^{-6}} = 18518.52` points.
    * In one millisecond, you would have approximately :math:`\frac{1}{54} = 18.52` points, roughly \(19\) points.

2. **Potential Action Potentials**

    * Max Number of Action Potentials: :math:`\frac{20176274}{93} \approx 216963`
    * Min Number of Action Potentials: :math:`\frac{20176274}{37} \approx 545307`

3. **Nyquist Frequency**

    * This would be :math:`\frac{18518.52}{2} = 9259.26` Hz.

4. **Data Volume**

    * Data Type: 32-bit
    * Array Size: :math:`20176274`

    To calculate the memory footprint, you would have:

    .. math::
        20176274 \times 32 \, \text{bits} = 645044768 \, \text{bits} = 80.63 \, \text{MB}
