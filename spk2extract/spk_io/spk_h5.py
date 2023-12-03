"""
Functions for saving/writing to h5 files using h5py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import tables

import spk2extract.extract
from spk2extract.logger import logger

import numpy as np


def _is_pyt_type(obj):
    """
    Check if a Python object is both regular and homogeneous for proper disk storage.

    Parameters:
    -----------
    obj : object
        Python object to check.

    Returns:
    --------
    bool
        True if the object is regular and homogeneous, False otherwise.
    """
    if isinstance(obj, np.ndarray) or np.isscalar(obj):
        return True
    if isinstance(obj, (list, tuple)):  # Check if obj is a native Python sequence
        if not obj:  # Empty sequence is fine
            return True

        # Check for homogeneity (all elements are of the same type)
        first_type = type(obj[0])
        if not all(isinstance(x, first_type) for x in obj):
            return False

        # Check for regularity (all sub-sequences should have the same length)
        if isinstance(obj[0], (list, tuple)):
            first_len = len(obj[0])
            if not all(len(x) == first_len for x in obj):
                return False
        return True
    return False


def save_channel_h5(fname: str, name: str, obj_list: list, metadata: dict):
    with tables.open_file(fname, mode="w") as h5file:
        file_group = h5file.create_group("/", name)
        file_group._v_attrs['metadata'] = metadata
        for channel in obj_list:
            _chan_arr_groups(h5file, file_group, channel)


def _chan_arr_groups(h5file: tables.file.File, parent_group: tables.Group, channel):

    # Create a group for each channel
    channel_group = h5file.create_group(parent_group, channel.name)
    channel_group._v_attrs['metadata'] = channel.metadata
    channel_group._v_attrs['type'] = channel.type
    channel_group._v_attrs['name'] = channel.name

    h5file.create_array(channel_group, 'data', channel.data)
    h5file.create_array(channel_group, 'times', channel.times)


def load_from_h5(h5file, group_name, cls):
    group = h5file.get_node("/", group_name)
    kwargs = {}
    for key in group._v_attrs._f_list():  # Load attributes
        kwargs[key] = group._v_attrs[key]
    for array in h5file.list_nodes(group, classname='Array'):  # Load arrays
        kwargs[array.name] = array.read()
    return cls(**kwargs)


def save_array_or_raise(group, key, value):
    if _is_pyt_type(value):
        group.create_array(group, key, value)
    else:
        raise ValueError(
            f"Value {value} of type {type(value)} is not compatible with pytables."
        )


def save_metadata_or_raise(group, metadata):
    for k, v in metadata.items():
        if _is_pyt_type(v):
            group._v_attrs[k] = v
        else:
            raise ValueError(
                f"Value {v} of type {type(v)} is not compatible with pytables."
            )


def save_event(h5file, event_group, event: spk2extract.extraction.Event):
    event_group_this = h5file.create_group(event_group, event.title, "Single Event")
    h5file.create_array(event_group_this, 'labels', event.labels)
    h5file.create_array(event_group_this, 'times', event.times)


def save_wave(h5file, wave_group, signal: spk2extract.extraction.Signal):
    wave_group_this = h5file.create_group(wave_group, signal.title, "Single Wave")
    h5file.create_array(wave_group_this, 'data', signal.data)
    h5file.create_array(wave_group_this, 'times', signal.times)


def write_h5(
        filename: Path | str,
        data: dict = None,
        events: Iterable = None,
        metadata_channel: dict = None,
        metadata_file: dict = None,
):
    """
    Creates a h5 file specific to the spike2 dataset.

    HDF5 Storage Structure
    ======================

    The HDF5 file is organized into groups and subgroups to store spike data, events, and metadata.

    Here is an overview of the storage structure:

    .. code-block:: text

        /
        ├── spikedata
        │   ├── Channel_1
        │   │   ├── spikes
        │   │   │   └── spikes_data (array)
        │   │   └── times
        │   │       └── times_data (array)
        │   ├── Channel_2
        │   │   ├── spikes
        │   │   │   └── spikes_data (array)
        │   │   └── times
        │   │       └── times_data (array)
        │   └── ...
        │
        ├── events
        │   └── events (array)
        │
        └── metadata
            ├── channel (group with attributes)
            └── (other metadata attributes)

    Parameters
    ----------
    filename : str or Path
        The filename to save to.
    data : dict or NamedTuple, optional
        A dictionary containing the data.
    events : list or np.ndarray, optional
        A list of events, where each event is a tuple of (event_name, event_time). Default is None.
    metadata_channel : dict, optional
        A dictionary containing metadata in dictionary form.
    metadata_file : dict, optional
        A dictionary containing the simple str metadata.

    Returns
    -------
    None

    """

    logger.info(f"Saving to {filename}...")
    if not Path(filename).parent.exists():
        logger.info(f"Creating directory {Path(filename).parent}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with tables.open_file(filename, mode="w") as h5file:
        wavedata_group = h5file.create_group("/", "spikedata", "Spike Data")
        metadata_group = h5file.create_group("/", "metadata", "Metadata")
        events_group = h5file.create_group("/", "events", "Event Data")

        if data is not None:
            logger.info("Saving data...")
            for dict_key, dict_value in data.items():
                channel_group = h5file.create_group(
                    wavedata_group, dict_key, f"{dict_key} Data"
                )
                if hasattr(dict_value, "items"):  # dict
                    for sub_type, sub_data in dict_value.items():
                        if sub_type == "spikes":
                            if _is_pyt_type(sub_data):
                                h5file.create_array(channel_group, sub_type, sub_data)
                            else:
                                raise ValueError(
                                    f"Value {sub_data} of type {type(sub_data)} is not compatible with pytables."
                                )
                        if sub_type == "times":
                            if _is_pyt_type(sub_data):
                                h5file.create_array(channel_group, sub_type, sub_data)
                            else:
                                raise ValueError(
                                    f"Value {sub_data} of type {type(sub_data)} is not compatible with pytables."
                                )
                else:
                    try:  # tuple
                        (spikes_data, times_data) = dict_value
                        h5file.create_array(channel_group, "spikes_data", spikes_data)
                        h5file.create_array(channel_group, "times_data", times_data)
                    except:
                        raise ValueError(
                            f"Value {dict_value} of type {type(dict_value)} is not compatible with h5py."
                        )

        if events is not None:
            logger.info("Saving events...")
            if hasattr(events, "items"):  # dict
                for sub_type, sub_data in events.items():
                    if _is_pyt_type(sub_data):
                        h5file.create_array(events_group, sub_type, sub_data)
            else:
                try:
                    for item in events:
                        if _is_pyt_type(item):
                            h5file.create_array(events_group, "event-data", item)
                except:
                    raise ValueError(
                        f"Value {events} of type {type(events)} is not compatible with this h5 writer."
                        f" Possible types are dict, namedtuple, or tuple with two elements:"
                        f"dict: {{'events': events_data, 'times': times_data}}"
                        f"namedtuple: namedtuple('events', ['events', 'times'])"
                        f"tuple: (events_data, times_data)"
                    )

        if metadata_file is not None:
            logger.info("Saving file metadata...")
            if not hasattr(metadata_file, "items"):
                raise ValueError(
                    f"The metadata_file parameter must be a dictionary, not:"
                    f"Type: {type(metadata_file)}."
                )
            try:
                for dict_key, value in metadata_file.items():
                    if _is_pyt_type(value):
                        if _is_pyt_type(value):
                            metadata_group._v_attrs[dict_key] = value
                        else:
                            raise ValueError(
                                f"Value {value} of type {type(value)} is not compatible with pytables."
                            )
            except Exception as e:
                raise ValueError(f"Saving metadata_file failed with error: {e}.")

        if metadata_channel is not None:
            logger.info("Saving channel metadata...")
            if not hasattr(metadata_channel, "items"):
                raise ValueError(
                    f"The metadata_channel parameter must be a dictionary, not:"
                    f"Type: {type(metadata_channel)}."
                )
            # store a dict for each channel
            for channel_name, channel in metadata_channel.items():
                channel_group = h5file.create_group(
                    metadata_group, channel_name, f"{channel_name} Metadata"
                )
                if hasattr(channel, "items"):
                    for dict_key, value in channel.items():
                        channel_group._v_attrs[dict_key] = value
                elif hasattr(channel, "_asdict"):
                    for dict_key, value in channel._asdict().items():
                        channel_group._v_attrs[dict_key] = value
                else:
                    raise ValueError(
                        f"Channel metadata must be a dictionary or namedtuple, not:"
                        f"Type: {type(channel)}."
                    )

    logger.info("Save complete.")
    return None


def __read_group(group: tables.Group) -> dict:
    """
    Read a single PyTables group and return a dictionary containing the data.

    Parameters
    ----------
    group : tables.Group
        PyTables group to read.

    Returns
    -------
    dict
        Dictionary containing the data from the PyTables group.
    """
    data = {}

    # attributes
    for attr_name in group._v_attrs._f_list("all"):
        attr_value = group._v_attrs[attr_name]
        data[attr_name] = attr_value

    # groups and leaves
    for node in group._f_iter_nodes():
        if isinstance(node, tables.Group):
            data[node._v_name] = __read_group(node)
        elif isinstance(node, tables.Array):
            array_data = node.read()
            if (
                    isinstance(array_data, np.ndarray) and array_data.dtype.kind == "S"
            ):  # Check if dtype is byte string
                array_data = array_data.astype(str)  # Convert to string
            elif isinstance(array_data, list):  # Handle list of byte strings
                array_data = [
                    item.decode() if isinstance(item, bytes) else item
                    for item in array_data
                ]
            data[node._v_name] = array_data

    return data


def read_h5(filename: str | Path) -> dict:
    """
    Read a single PyTables HDF5 file and return a dictionary containing the data.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary containing the data from the HDF5 file.
    """
    with tables.open_file(str(filename), "r") as h5file:
        data = __read_group(h5file.root)
    return data


def get_h5_filename(file_dir):
    if "SHH_CONNECTION" in os.environ:
        shell = True

    file_list = os.listdir(file_dir)
    h5_files = [f for f in file_list if f.endswith(".h5")]
    return os.path.join(file_dir, h5_files[0])

