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
from pathlib import Path

from numpy import __version__ as numpyversion
from packaging.version import Version
from platformdirs import *

import logger
import directory_manager
import main
import sorter
import spk_config
import utils
from .. import spk_io

# TODO: Add version check
__version__ = "0.1.0"

version = __version__
__name__ = "clustersort"
__author__ = "Flynn OConnell"

sortdir = Path().home() / "cpsort"
if not sortdir.exists():
    sortdir.mkdir(exist_ok=True)

cache_dir = user_cache_dir(__name__,)  # Cache, temp files
config_dir = user_config_dir(__name__,)  # Config, parameters and options
log_dir = user_log_dir(__name__,)  # Logs, .log files primarily

if Version(numpyversion) >= Version("1.24.0"):
    raise ImportError(
        "numpy version 1.24.0 or greater is not supported due to numba incompatibility, "
        "please downgrade to 1.23.5 or lower"
    )
