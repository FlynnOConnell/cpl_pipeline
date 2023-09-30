==================
spk2extract
==================

.. image:: https://readthedocs.org/projects/spk2extract/badge/?version=latest
    :target: https://spk2extract.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

Data extraction pipeline for spike2 electrophysiological data.
This primarily relies on the [CED](https://ced.co.uk/) [SonPy](https://github.com/divieira/sonpy) library, which
is ill-documented and is a thin C++ wrapper around code that we will never see.

Understanding the Spike2 Data Types
-----------------------------------
We are working with data from an Analog-to-Digital Converter (ADC). A bit about what this all means:

ADC (Analog-to-Digital Converter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ADC converts continuous analog signals (like electrical voltage) into a digital representation. The digital representation is often stored in bits, and in this case, it's stored in 16 bits, which means each measurement (or sample) is represented with 16 bits of data. This gives you a range of possible values from -32768 to 32767, which are derived from :math:`2^{16}` different possible 16-bit values (ranging from 0 to 65535) but offset to allow for negative values.

Waveform Data
^^^^^^^^^^^^^

This term refers to data representing a wave acquired over time, and it is time-continuous, meaning that it has been sampled at regular intervals over time, forming a continuous signal.

Scale and Offset
^^^^^^^^^^^^^^^^^

Each ADC channel contains a scale and offset that allows you to transform the recorded integer values into a different range according to the linear transformation equation:

.. math:: y = mx + c

where:

- :math:`y` is the transformed data value
- :math:`m` is the scale factor
- :math:`x` is the original data value
- :math:`c` is the offset

This transformation allows you to calibrate the data to represent real-world quantities accurately. In the case of Spike2, the scale and offset would be used to convert the ADC counts to a voltage value based on the specifications of the Spike2 hardware used to acquire the data.

Working with this Data
^^^^^^^^^^^^^^^^^^^^^^^

Considering the bit-depth of your signals (16-bit), it allows for a decent range of discrete values, giving us a high-resolution representation of the continuous signals that were measured.
