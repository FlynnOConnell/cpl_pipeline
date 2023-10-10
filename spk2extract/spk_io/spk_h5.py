"""
Functions for saving/writing to h5 files using h5py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Any

import h5py
import tables

from spk2extract.spk_io.utils import is_h5py_compatible

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

def write_complex_h5(
    filename: Path | str,
    data: dict = None,
    events: Iterable = None,
    metadata_file: dict = None,
    metadata_channel: dict[Any] = None,
):
    """
    Creates a h5 file specific to the spike2 dataset.

    There is a group for metadata, metadata_dict, data, and sampling_rates.

    .. note::

        The metadata_file group contains metadata for the entire file, i.e. the bandpass filter frequencies.
        The metadata_channel group contains metadata for each channel, i.e. the channel type, sampling rate.
        Data contains the actual data, i.e. the waveforms and spike times.

    Parameters
    ----------
    filename : str or Path
        The filename to save to.
    data : dict or NamedTuple, optional
        A dictionary containing the data.
    events : list or np.ndarray, optional
        A list of events, where each event is a tuple of (event_name, event_time). Default is None.
    metadata_file : dict, optional
        A dictionary containing the simple str metadata.
    metadata_channel : dict, optional
        A dictionary containing the metadata dictionaries.

    Returns
    -------
    None

    """
    if not Path(filename).parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as f:
        # 1) Save data dictionaries
        if data is not None:
            data_grp = f.create_group("data")
            for data_key, data_value in data.items():
                sub_group = data_grp.create_group(data_key)
                # Convert the namedtuple to a dict to store in the h5 file as a subgroup
                # The leading _ in _asdict() is to prevent named conflicts, not to indicate privacy
                # See: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict
                if isinstance(data_value, dict):
                    for field_name, field_value in data_value.items():
                        if is_h5py_compatible(field_value):
                            sub_group.create_dataset(field_name, data=field_value)
                        else:
                            raise ValueError(
                                f"Value {field_value} of type {type(field_value)} is not compatible with h5py."
                            )
                else:
                    for field_name, field_value in data_value._asdict().items():
                        if is_h5py_compatible(field_value):
                            sub_group.create_dataset(field_name, data=field_value)
                        else:
                            raise ValueError(
                                f"Value {field_value} of type {type(field_value)} is not compatible with h5py."
                            )

        # 2) Save events
        if events is not None:
            event_grp = f.create_group("events")
            # keep this as a group so other groups / events can be added if needed
            if is_h5py_compatible(events):
                event_grp.create_dataset(field_name, data=events)
            else:
                raise ValueError(
                    f"Value {events} of type {type(events)} is not compatible with h5py."
                )

        # 3) Save metadata for the entire file (any simple type supported by HDF5)
        if metadata_file is not None:
            metadata_file_grp = f.create_group("metadata_file")
            for key, value in metadata_file.items():
                if isinstance(value, dict):
                    sub_group = metadata_file_grp.create_group(key)
                    for sub_key, sub_value in value.items():
                        str_key = (
                            str(sub_key)
                            if not isinstance(sub_key, tuple)
                            else ",".join(map(str, sub_key))
                        )
                        sub_group.attrs[str_key] = sub_value
                else:
                    metadata_file_grp.attrs[key] = value

        # 4) Save metadata per-channel
        if metadata_channel is not None:
            metadata_channel_grp = f.create_group("metadata_channel")
            for dict_name, dict_data in metadata_channel.items():
                sub_group = metadata_channel_grp.create_group(dict_name)
                for key, value in dict_data.items():
                    str_key = (
                        str(key) if not isinstance(key, tuple) else ",".join(map(str, key))
                    )
                    sub_group.attrs[str_key] = value

    return None

def create_empty_data_h5(filename, overwrite=False, shell=False):
    """
    Create empty h5 store for blech data with approriate data groups

    Parameters
    ----------
    filename : str, absolute path to h5 file for recording
    """
    if 'SHH_CONNECTION' in os.environ:
        shell = True

    if not filename.endswith('.h5') and not filename.endswith('.hdf5'):
        filename += '.h5'

    basename = os.path.splitext(os.path.basename(filename))[0]

    if os.path.isfile(filename):
        os.remove(filename)
        print('Done!')

    print('Creating empty HDF5 store with raw data groups')
    data_groups = ['raw', 'unit', 'lfp']
    with tables.open_file(filename, 'w', title=basename) as hf5:
        for grp in data_groups:
            hf5.create_group('/', grp)
        hf5.flush()
    return filename


def create_hdf_arrays(file_name, num_channels, overwrite=False):
    if os.path.isfile(file_name):
        if not overwrite:
            print(f"{file_name} already exists. Exiting.")
            return
        else:
            os.remove(file_name)

    print('Creating empty HDF5 store with raw data groups...')
    data_groups = ['unit', 'lfp']
    atom = tables.IntAtom()
    f_atom = tables.Float64Atom()

    with tables.open_file(file_name, 'w') as hf5:
        for grp in data_groups:
            hf5.create_group('/', grp)

        # Create array for raw time vector
        hf5.create_earray('/raw', 'time_vector', f_atom, (0,))

        # Create arrays for each channel
        for i in range(1, num_channels + 1):
            hf5.create_earray('/raw', f'channel_{i}', atom, (0,))
    print('Done!')

def get_h5_filename(file_dir, shell=True):
    """
    """
    if 'SHH_CONNECTION' in os.environ:
        shell = True

    file_list = os.listdir(file_dir)
    h5_files = [f for f in file_list if f.endswith('.h5')]
    return os.path.join(file_dir, h5_files[0])

def read_files_into_arrays(file_name, time_data, channel_data):
    if not Path(file_name).exists():
        print(f"{file_name} does not exist. Exiting.")
        return

    print(f'Appending data to {file_name}...')

    with tables.open_file(file_name, 'r+') as hf5:
        hf5.root.raw.time_vector.append(time_data)
        # Append channel data
        for i, ch_data in enumerate(channel_data, start=1):
            hf5.get_node(f'/raw/channel_{i}').append(ch_data)
    print('Done!')