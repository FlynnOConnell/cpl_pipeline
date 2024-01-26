==========================
cpl_pipeline Documentation
==========================



.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: gettingstarted

    cpl_pipeline

.. _quickstart:

cpl_pipeline is heavily adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017) [1]_.
- code found at `bleckpy on github<https://github.com/nubs01/blechpy>`_.

Quickstart
==========

For more complete installation instructions, see :ref:`install <install>`.

1. Using pip:

.. code-block:: bash

    # clone the repository
    git clone https://github.com/FlynnOConnell/cpl_pipeline.git
    cd cpl_pipeline

    # install the package with pip
    pip install -e .

    # OR install via setup.py
    python setup.py install

2. Conda/Mamba (coming soon)

.. code-block:: bash

    conda install cpl_pipeline

=============================
Semi-Supervised Spike Sorting
=============================

This Python repository is adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017) [1]_.
A large percentage of the clustering parameters were made in reference to Reddish (2005) [2]_.

.. note::

   This program is designed for sorting spikes from electrophysiological recordings into single, isolated units. The primary input is a .h5 file containing the continuous signal or thresholded waveforms.

Sorting Criteria
----------------
The primary criteria for considering a unit isolated are:

#. 1 ms ISIs must be <= 0.5%
#. The waveform must be cellular
#. The unit must be sufficiently separated based on Mahalanobis distribution
#. L-Ratio must be <= 0.1, as described in Schmitzer-Torbert et al. (2005) [2]_.


Post-Processing
---------------

The post-processing is carried out via a GUI.

.. note::

   This step requires both the .h5 files and the output folders from the Processing step.

References
==========

.. [1] Mukherjee, Narendra & Wachutka, Joseph & Katz, Donald. (2017). Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents. 98-105.

.. [2] Schmitzer-Torbert N, Jackson J, Henze D, Harris K, Redish AD. Quantitative measures of cluster quality for use in extracellular recordings. Neuroscience. 2005;131:1–11.

External Resources
==================

For interacting with Spike2 data, the SonPy library is used and available via `pypi.org <https://pypi.org/project/sonpy/>`_.

.. _contact:

Contact
=======

- GitHub: https://github.com/flynnoconnell/cpl_pipeline
- Email: Flynnoconnell@gmail.com
