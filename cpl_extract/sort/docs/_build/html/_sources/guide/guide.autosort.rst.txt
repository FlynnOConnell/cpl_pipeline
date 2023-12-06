=============================
Semi-Supervised Spike Sorting
=============================

This Python repository is adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017) [1]_.
A large percentage of the clustering parameters were made in reference to Reddish (2005) [2]_.

.. note::

   This program is designed for sorting spikes from electrophysiological recordings into single, isolated units. The primary input is a .h5 file containing the continuous signal or thresholded waveforms.

**Compatibility and Licensing**

This software is compatible with Windows, macOS, and Linux, and is well-suited for containerization and high-performance computing clusters. It is distributed under the GNU General Public License v3.0 (GPLv3). For more information, consult the LICENSE file in this repository.

Usage
=====

Analysis
--------

The primary folder used for analysis is the 'superplots' folder. The 'Plots' folder contains individual plots, but in 'superplots', they are compiled for user convenience.

.. note::

   Important files include `.info`, which contains information about the sort run, and `clustering_results_compiled_isoi.xlsx`, which contains details about each cluster.


Criteria
---------
The primary criteria for considering a unit isolated are:

#. 1 ms ISIs must be <= 0.5%
#. The waveform must be cellular
#. The unit must be sufficiently separated based on Mahalanobis distribution
#. L-Ratio must be <= 0.1, as described in Schmitzer-Torbert et al. (2005) [2]_.

.. note::

   For structuring the plot paths and further details on L-Ratio, refer to the Autosort configuration file.

Post-Processing
---------------

The post-processing is carried out via a GUI.

.. note::

   This step requires both the .h5 files and the output folders from the Processing step.

Pipeline
========

The pipeline functions as follows:

1. **Pre-Processing**: Data is extracted from Spike2 files and packaged into .h5 files.
2. **Processing**: Multiple steps are performed, as detailed in Mukherjee et al. (2017) [1]_.
3. **Post-Processing**: The data is packaged into .json files.

References
==========

.. [1] Mukherjee, Narendra & Wachutka, Joseph & Katz, Donald. (2017). Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents. 98-105.

.. [2] Schmitzer-Torbert N, Jackson J, Henze D, Harris K, Redish AD. Quantitative measures of cluster quality for use in extracellular recordings. Neuroscience. 2005;131:1–11.

External Resources
==================

For interacting with Spike2 data, the SonPy library is used and available via `pypi.org <https://pypi.org/project/sonpy/>`_.


