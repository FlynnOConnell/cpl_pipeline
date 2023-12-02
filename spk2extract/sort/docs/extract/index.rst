=========================
spk2extract Documentation
=========================

Clustersort is a python utility for extracting spike waveforms from raw Spike2 data files. It gives
some more transparency to what exactly the SonPy module is doing when it extracts spikes, and provides
some utilities for extracting thresholded spikes, dejittering and cleaning up the data, and saving the resulting
data to hdf5 files.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: gettingstarted

   Installation <install>
   Usage Guide <usage>
   Api <api/index>
   Preprocessing <preprocessing>

.. _quickstart:

Quickstart
==========

Install
-------

For more complete installation instructions, see :ref:`install <install>`.

1. Using pip:

.. code-block:: bash

    pip install spk2extract

2. Conda/Mamba (coming soon)

.. code-block:: bash

    conda install spk2extract

3. From source:

.. code-block:: bash

    git clone https://github.com/FlynnOConnell/spk2extract.git
    cd spk2extract
    python setup.py install


Usage
-----

For more complete usage instructions, see :ref:`usage <usage>`.
For more complete api instructions, see :ref:`api <api/index>`.

.. code-block::

    import spk2extract
    from pathlib import Path

    my_smr_file = Path().home() / 'data' / 'my_smr_file.smr'
    spike_data = spk2extract.SpikeData(my_smr_file)

.. _contribute:

Contribute
==========

1. Fork the repository.
2. Create your feature bra  nch.
3. Commit your changes.
4. Open a pull request.

.. _contact:

Contact
=======

- GitHub: https://github.com/flynnoconnell/spk2extract
- Email: Flynnoconnell@gmail.com
