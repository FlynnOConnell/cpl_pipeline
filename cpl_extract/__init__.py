import os

import numpy as np

from . import utils, analysis
from .spk_io import *
from .utils import *
from .analysis import *
from .base import *

from .base.objects import load_project, load_experiment
from .base.experiment import experiment
from .base.project import project


def pad_arrays_to_same_length(arr_list, max_diff=100):
    """
    Pads numpy arrays to the same length.

    Parameters:
    - arr_list (list of np.array): The list of arrays to pad
    - max_diff (int): Maximum allowed difference in lengths

    Returns:
    - list of np.array: List of padded arrays
    """
    lengths = [len(arr) for arr in arr_list]
    max_length = max(lengths)
    min_length = min(lengths)

    if max_length - min_length > max_diff:
        raise ValueError("Arrays differ by more than the allowed maximum difference")

    padded_list = []
    for arr in arr_list:
        pad_length = max_length - len(arr)
        padded_arr = np.pad(arr, (0, pad_length), "constant", constant_values=0)
        padded_list.append(padded_arr)

    return padded_list

def extract_common_key(filepath):
    parts = filepath.stem.split("_")
    return "_".join(parts[:-1])

__name__ = "cpl_extract"
__author__ = "Flynn OConnell"

__all__ = [
    "spk_io",
    "utils",
    "analysis",
    "Dataset",
    "load_params",
    "load_dataset",
    "load_pickled_object",
    "load_data",
    "load_project",
    "load_experiment",
    "detect_number_of_cores",
    "experiment",
    "project",
]

# Version
version = "0.0.2"

class CacheDict(dict):
    """
    A dictionary that prevents itself from growing too much.
    """

    def __init__(self, maxentries):
        self.maxentries = maxentries
        super().__init__(self)

    def __setitem__(self, key, value):
        # Protection against growing the cache too much
        if len(self) > self.maxentries:
            # Remove a 10% of (arbitrary) elements from the cache
            entries_to_remove = self.maxentries / 10
            for k in list(self)[:entries_to_remove]:
                super().__delitem__(k)
        super().__setitem__(key, value)

def detect_number_of_cores():
    """Detects the number of cores on a system."""

    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default

def _test():
    """Run ``doctest``"""
    import doctest
    doctest.testmod()
