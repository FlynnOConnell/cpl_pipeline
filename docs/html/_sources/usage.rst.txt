.. _usage:

Usage
=====

The workhorse of this package is the :class:`Spike2Data <cpl_extract.Spike2Data>` class. It contains properties for loading
and extracting all of the data from .smr files.

The :class:`SpikeData <cpl_pipeline.SpikeData>` class is initialized with a path to a .smr file.

The data is NOT loaded into memory when the :class:`SpikeData <cpl_pipeline.SpikeData>` object is created,
this only happens when SpikeData.process() is called. The data can then be accessed using the properties of the the class.
See the documentation for :class:`SpikeData <cpl_pipeline.SpikeData>` for more information.

.. code-block::

    import cpl_pipeline
    from pathlib import Path

    my_smr_file = Path().home() / 'data' / 'my_smr_file.smr'
    spike_data = cpl_pipeline.SpikeData(my_smr_file)
    spike_data.process()


You may also wish to iterate through a directory containing many .smr files:

.. code-block::

    import cpl_pipeline
    from pathlib import Path

    smr_path = Path().home() / "data" / "smr"
    files = [f for f in smr_path.glob("*.smr")]
    all_data = {}

    for file in files:
        data = cpl_pipeline.SpikeData(file)
        data.process()
        all_data[file.stem] = data
