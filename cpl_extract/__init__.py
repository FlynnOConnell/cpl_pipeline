#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import psutil

try:
    from platformdirs import user_cache_dir, user_config_dir, user_log_dir
except ImportError:
    user_cache_dir = None
    user_config_dir = None
    user_log_dir = None
    pass
from pathlib import Path
from . import spk_io, utils, analysis


__name__ = "cpl_extract"
__author__ = "Flynn OConnell"
__all__ = [
    "spk_io",
    "utils",
    "analysis",
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

def calculate_optimal_chunk_size(item_size_bytes, memory_usage_fraction=0.5):
    """Calculates the optimal chunk size for a given item size."""
    return int(
        (psutil.virtual_memory().available * memory_usage_fraction) / item_size_bytes
    )

def _test():
    """Run ``doctest``"""
    import doctest
    doctest.testmod()

def check_substring_content(main_string, substring) -> bool:
    """Checks if any combination of the substring is in the main string."""
    return substring.lower() in main_string.lower()

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

# Platform-dependent directories
def _init_directories():
    cpe_dir = Path().home() / "cpl_extract"
    cpe_dir.mkdir(exist_ok=True)

    # use pre-existing dirs if possible
    if user_cache_dir and user_config_dir and user_log_dir:
        return {
            "cache_dir": user_cache_dir(__name__),
            "config_dir": user_config_dir(__name__),
            "log_dir": user_log_dir(__name__),
            "cpe_dir": cpe_dir,
        }
    else:
        return {
            "cpe_dir": cpe_dir,
            "cache_dir": cpe_dir / "cache",
            "config_dir": cpe_dir / "config",
            "log_dir": cpe_dir / "logs",
        }

directories = _init_directories()
__all__.extend(list(directories.keys()))
globals().update(directories)
