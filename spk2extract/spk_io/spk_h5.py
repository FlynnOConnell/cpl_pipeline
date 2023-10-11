"""
Functions for saving/writing to h5 files using h5py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import tables

def write_h5(
    filename: Path | str,
    data: dict = None,
    events: Iterable = None,
    metadata_file: dict = None,
    metadata_channel: dict = None,
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
    metadata_file : dict, optional
        A dictionary containing the simple str metadata.
    metadata_channel : dict, optional
        A dictionary containing metadata in dictionary form.

    Returns
    -------
    None

    """

    if not Path(filename).parent.exists():
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with tables.open_file(filename, mode="w") as h5file:
        wavedata_group = h5file.create_group("/", "spikedata", "Spike Data")
        metadata_group = h5file.create_group("/", "metadata", "Metadata")
        events_group = h5file.create_group("/", "events", "Event Data")

        if data is not None:
            for dict_key, dict_value in data.items():
                channel_group = h5file.create_group(
                    wavedata_group, dict_key, f"{dict_key} Data"
                )
                spikes_group = h5file.create_group(
                    channel_group, "spikes", "Spikes Data"
                )
                times_group = h5file.create_group(channel_group, "times", "Times Data")

                if hasattr(dict_value, "_asdict"):  # namedtuple
                    for field_name, field_value in dict_value._asdict().items():
                        if field_name == "spikes":
                            h5file.create_array(spikes_group, field_name, field_value)
                        if field_name == "times":
                            h5file.create_array(times_group, field_name, field_value)

                elif hasattr(dict_value, "items"):  # dict
                    for sub_type, sub_data in dict_value.items():
                        if sub_type == "spikes":
                            h5file.create_array(spikes_group, sub_type, sub_data)
                        if sub_type == "times":
                            h5file.create_array(times_group, sub_type, sub_data)
                else:
                    try:
                        (spikes_data, times_data) = dict_value
                        h5file.create_array(spikes_group, "spikes_data", spikes_data)
                        h5file.create_array(times_group, "times_data", times_data)
                    except:
                        raise ValueError(
                            f"Value {dict_value} of type {type(dict_value)} is not compatible with h5py."
                        )

        if events is not None:
            h5file.create_array(events_group, "events", events)

        if metadata_file is not None:
            for dict_key, value in metadata_file.items():
                metadata_group._v_attrs[dict_key] = value

        if metadata_channel is not None:
            channel_group = h5file.create_group(
                metadata_group, "channel", "Channel Metadata"
            )
            for dict_key, value in metadata_channel.items():
                channel_group._v_attrs[dict_key] = value
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

    # Reading attributes
    for attr_name in group._v_attrs._f_list("all"):
        attr_value = group._v_attrs[attr_name]
        data[attr_name] = attr_value

    # Reading child nodes (groups and leaves)
    for node in group._f_iter_nodes():
        if isinstance(node, tables.Group):
            data[node._v_name] = __read_group(node)
        elif isinstance(node, tables.Array):
            data[node._v_name] = node.read()

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
