"""

``clustersort``
===============

*clustersort. A fully automated and configurable spike sorting pipeline for extracellular recordings.*

A utility for cluster-analysis spike sorting with electrophysiological data.
Once spikes are sorted, users are able to post-process the spikes using
plots for mahalanobis distance, ISI, and autocorrelograms. The spike data can also be exported to
a variety of formats for further analysis in other programs.

Pipeline
--------

1. Read in the data from a h5, npy, or nwb file.
2. Cluster the spikes.
3. Perform breach analysis on the clusters.
4. Resort the clusters based on the breach analysis.
5. Save the data to an HDF5 file, and graphs to given plotting folders.

Documentation Guide
-------------------

I recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.

"""
from . import directory_manager
from . import main
from . import sorter
from . import spk_config
from . import utils

__all__ = ["directory_manager", "main", "sorter", "spk_config", "utils"]