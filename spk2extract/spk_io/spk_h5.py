"""
Functions for saving/writing to h5 files using h5py.
"""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


def write_h5(filename: str, unit_dict: dict):
    """
    Save the specified data dictionaries to an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the output HDF5 file.
    unit_dict : dict
        Dictionary containing unit data.

    Examples
    --------
    >>> write_h5('output.h5', {'unit1': [1, 2, 3], 'unit2': [4, 5, 6]})
    """
    with h5py.File(filename, 'w') as f:
        unit_group = f.create_group('unit')
        for key, data in unit_dict.items():
            unit_group.create_dataset(key, data=data)


def write_complex_h5(filename, metadata, metadata_dicts, data):
    with h5py.File(filename, "w") as f:
        # Save metadata (any simple type supported by HDF5)
        metadata_grp = f.create_group("metadata")
        for key, value in metadata.items():
            metadata_grp.attrs[key] = value

        # Save metadata dictionaries
        metadata_dict_grp = f.create_group("metadata_dicts")
        for dict_name, dict_data in metadata_dicts.items():
            sub_group = metadata_dict_grp.create_group(dict_name)
            for key, value in dict_data.items():
                str_key = (
                    str(key) if not isinstance(key, tuple) else ",".join(map(str, key))
                )
                sub_group.attrs[
                    str_key
                ] = value  # this value can be int, float, str, etc.

        # Save data dictionaries
        data_grp = f.create_group("data")
        for data_key, data_value in data.items():
            str_key = ",".join(map(str, data_key))
            data_grp.create_dataset(str_key, data=np.array(data_value))

def __read_group(group: h5py.Group) -> dict:
    """
    Read a single HDF5 group and return a dictionary containing the data.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to read.

    Returns
    -------
    dict
        Dictionary containing the data from the HDF5 group.
    """
    data = {}
    for attr_name, attr_value in group.attrs.items():
        data[attr_name] = attr_value
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = __read_group(item)
        elif isinstance(item, h5py.Dataset):
            data[key] = item[()]
    return data


def read_h5(filename: str | Path) -> dict:
    """
    Read a single HDF5 file and return a dictionary containing the data.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary containing the data from the HDF5 file.

    Examples
    --------
    >>> data = read_h5('input.h5')
    """
    with h5py.File(filename, "r") as f:
        data = __read_group(f)
    return data
