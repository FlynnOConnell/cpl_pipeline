.. _usage:

Usage
=====

The workhorse of this package is the :class:`SpikeData <spk2extract.SpikeData>` class. It contains properties for loading
and extracting all of the data from .smr files.

The :class:`SpikeData <spk2extract.SpikeData>` class is initialized with a path to a .smr file.

The data is NOT loaded into memory when the :class:`SpikeData <spk2extract.SpikeData>` object is created,
this only happens when SpikeData.process() is called. The data can then be accessed using the properties of the the class.
See the documentation for :class:`SpikeData <spk2extract.SpikeData>` for more information.

.. code-block::

    import spk2extract
    from pathlib import Path

    my_smr_file = Path().home() / 'data' / 'my_smr_file.smr'
    spike_data = spk2extract.SpikeData(my_smr_file)
