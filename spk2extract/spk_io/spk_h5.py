"""
Functions for saving/writing to h5 files using h5py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import tables
from spk2extract.logs import logger

import numpy as np


def is_compatible_type(obj):
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


def save_array_or_raise(group, key, value):
    if is_compatible_type(value):
        group.create_array(group, key, value)
    else:
        raise ValueError(
            f"Value {value} of type {type(value)} is not compatible with pytables."
        )


def save_metadata_or_raise(group, metadata):
    for k, v in metadata.items():
        if is_compatible_type(v):
            group._v_attrs[k] = v
        else:
            raise ValueError(
                f"Value {v} of type {type(v)} is not compatible with pytables."
            )


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
                if hasattr(dict_value, "_asdict"):  # namedtuple
                    for field_name, field_value in dict_value._asdict().items():
                        if field_name == "spikes":
                            if is_compatible_type(field_value):
                                h5file.create_array(
                                    channel_group, field_name, field_value
                                )
                            else:
                                raise ValueError(
                                    f"Value {field_value} of type {type(field_value)} is not compatible with pytables."
                                )
                        if field_name == "times":
                            if is_compatible_type(field_value):
                                h5file.create_array(
                                    channel_group, field_name, field_value
                                )
                            else:
                                raise ValueError(
                                    f"Value {field_value} of type {type(field_value)} is not compatible with pytables."
                                )

                elif hasattr(dict_value, "items"):  # dict
                    for sub_type, sub_data in dict_value.items():
                        if sub_type == "spikes":
                            if is_compatible_type(sub_data):
                                h5file.create_array(channel_group, sub_type, sub_data)
                            else:
                                raise ValueError(
                                    f"Value {sub_data} of type {type(sub_data)} is not compatible with pytables."
                                )
                        if sub_type == "times":
                            if is_compatible_type(sub_data):
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
            if hasattr(events, "_asdict"):  # namedtuple
                for field_name, field_value in events._asdict().items():
                    if is_compatible_type(field_value):
                        h5file.create_array(events_group, field_name, field_value)
            elif hasattr(events, "items"):  # dict
                for sub_type, sub_data in events.items():
                    if sub_type == "events" and is_compatible_type(sub_data):
                        h5file.create_array(events_group, sub_type, sub_data)
            else:
                try:
                    for item in events:
                        if is_compatible_type(item):
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
                    if is_compatible_type(value):
                        if is_compatible_type(value):
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


def read_files_into_arrays(file_name, time_data, channel_data):
    if not Path(file_name).exists():
        print(f"{file_name} does not exist. Exiting.")
        return

    print(f"Appending data to {file_name}...")

    with tables.open_file(file_name, "r+") as hf5:
        hf5.root.raw.time_vector.append(time_data)
        # Append channel data
        for i, ch_data in enumerate(channel_data, start=1):
            hf5.get_node(f"/raw/channel_{i}").append(ch_data)
    print("Done!")


if __name__ == "__main__":
    # get all files with "pre" in the name before .h5
    path = Path().home() / "spk2extract" / "h5"
    pre_files = list(path.glob("*pre*.h5"))
    post_files = list(path.glob("*post*.h5"))
    testdata = read_h5(pre_files[0])
    combined_metadata = {}
