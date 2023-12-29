from __future__ import annotations

import pprint
from pathlib import Path

import tables
import re
import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from icecream import ic

from cpl_extract.analysis import cluster as clust
from cpl_extract.spk_io import userio, println, paramio
from cpl_extract.utils import particles

SUPP_REC_TYPES = {
    "spike230bit": ".smr",
    "spike264bit": ".smrx",
    "plexon": ".pl2",
}

DATA_GROUPS = ["raw", "raw_lfp", "time", "digital_in", "digital_out", "trial_info"]


def merge_h5_files(file_list: list[str | Path]):
    """
    Merges HDF5 files created by create_hdf_arrays function.
    If the process fails, the created file is deleted.
    """

    file_list = [str(f) for f in file_list]
    new_h5_path = Path(file_list[0]).parent / "merged_file.h5"

    try:
        with tables.open_file(file_list[0], 'r') as h_1, tables.open_file(file_list[1], 'r') as h_2:
            with tables.open_file(new_h5_path, 'w') as new_h5:
                # Merge groups /raw, /raw_lfp, /digital_in, /digital_out
                for group_name in ['/raw', '/raw_lfp', '/digital_in', '/digital_out']:
                    if group_name in h_1.root and group_name in h_2.root:
                        h1_group = getattr(h_1.root, group_name.strip('/'))
                        h2_group = getattr(h_2.root, group_name.strip('/'))

                        # Create group in new file
                        new_h5.create_group('/', group_name.strip('/'), group_name.strip('/').title())

                        # Verify and copy attributes
                        for attr in h1_group._v_attrs._f_list():
                            assert h1_group._v_attrs[attr] == h2_group._v_attrs[
                                attr], f"Attribute {attr} mismatch in group {group_name}"
                            getattr(new_h5.root, group_name.strip('/'))._v_attrs[attr] = h1_group._v_attrs[attr]

                        # Merge arrays within each group
                        for node in h1_group._f_list_nodes():
                            node_name = node._v_name
                            combined_array = np.concatenate([node[:], getattr(h2_group, node_name)[:]])
                            new_h5.create_earray(group_name, node_name, obj=combined_array)

        print(f"Merged file created at: {new_h5_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Check if the file exists and if any exception occurred, then delete the file
        if new_h5_path.exists():
            try:
                new_h5_path.unlink()
                print(f"Partial file deleted: {new_h5_path}")
            except Exception as e:
                print(f"Error deleting partial file: {e}")


def get_h5_filename(file_dir, shell=True, merge=False):
    """
    Return the name of the h5 file found in file_dir.
    Asks for selection if multiple found

    Parameters
    ----------
    file_dir : str, path to recording directory
    shell : bool (optional)
        True (default) for command line interface if multiple h5 files found
        False for GUI
    merge : bool (optional)
        True to return list of all h5 files in directory

    Returns
    -------
    str
        filename of h5 file in directory (not full path), None if no file found
    """

    if "SHH_CONNECTION" in os.environ:
        shell = True

    h5_files = list(Path(file_dir).glob("*.h5"))
    if len(h5_files) > 1:
        if merge:
             return [os.path.join(file_dir, h5_files[n]) for n in range(len(h5_files))]
        else:
            choice = userio.select_from_list(
                "Choose which h5 file to load",
                h5_files,
                "Multiple h5 stores found",
                shell=shell,
            )
            if choice is None:
                return None
            else:
                h5_files = [choice]
    elif len(h5_files) == 0:
        return None
    return os.path.join(file_dir, h5_files[0])


def get_unit_descriptor(rec_dir, unit_num, h5_file=None):
    """Returns the unit description for a unit in the h5 file in rec_dir"""
    if isinstance(unit_num, str):
        unit_num = parse_unit_number(unit_num)

    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    with tables.open_file(h5_file, "r") as hf5:
        descrip = hf5.root.unit_descriptor[unit_num]

    return descrip


def get_unit_names(rec_dir, h5_file=None):
    """Finds h5 file in dir, gets names of sorted units and returns

    Parameters
    ----------
    rec_dir : str, full path to recording dir

    Returns
    -------
    list of str
    """
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    with tables.open_file(h5_file, "r") as hf5:
        if "/sorted_units" in hf5:
            units = hf5.list_nodes("/sorted_units")
            unit_names = [x._v_name for x in units]
        else:
            unit_names = []

    return unit_names


def get_unit_table(rec_dir, h5_file=None):
    """Returns pandas DataFrame with sorted unit info read from hdf5 store

    Parameters
    ----------
    rec_dir : str, path to dir containing .h5 file

    Returns
    -------
    pandas.DataFrame with columns:
        - unit_name
        - unit_num
        - electrode
        - single_unit
        - regular_spiking
        - fast_spiking
    """
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    units = get_unit_names(rec_dir, h5_file=h5_file)
    unit_table = pd.DataFrame(units, columns=["unit_name"])
    unit_table["unit_num"] = unit_table["unit_name"].apply(
        lambda x: parse_unit_number(x)
    )

    def add_descrip(row):
        descrip = get_unit_descriptor(rec_dir, row["unit_num"], h5_file=h5_file)
        row["electrode"] = descrip["electrode_number"]
        row["single_unit"] = bool(descrip["single_unit"])
        row["multi_unit"] = bool(descrip["multi_unit"])
        return row

    unit_table = unit_table.apply(add_descrip, axis=1)
    return unit_table


def get_next_unit_name(rec_dir, h5_file=None):
    """
    returns node name for next sorted unit

    Parameters
    ----------
    rec_dir : str, full path to recording directory

    Returns
    -------
    str , name of next unit in sequence ex: "unit001"
    """
    units = get_unit_names(rec_dir, h5_file=h5_file)
    unit_nums = sorted([parse_unit_number(i) for i in units])
    if not units:
        out = "unit%03d" % 0
    else:
        out = "unit%03d" % int(max(unit_nums) + 1)

    return out


def get_spike_data(rec_dir, units=None, din=None, trials=None, h5_file=None):
    """
    Opens hf5 file in rec_dir and returns a Trial x Time spike array and a
    1D time vector

    Parameters
    ----------
    rec_dir : str, path to recording directory
    units : str or int or list of str/int, unit names or unit numbers
    din : int, digital input channel
    trials: int or list-like
        if None (default), returns all trials, if int N returns first N-trials
        for each din, if list-like then returns those indices for each taste

    Returns
    -------
    time : numpy.array
    spike_array : numpy.array
    """
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    if units is None:
        ic("No units specified, retreiving all units", rec_dir)
        units = get_unit_names(rec_dir)
    elif not isinstance(units, list):
        units = [units]

    unit_nums = []
    for u in units:
        if isinstance(u, int):
            unit_nums.append(u)
        else:
            unit_nums.append(parse_unit_number(u))

    unit_nums = np.array(unit_nums)
    if len(unit_nums) == 1:
        unit_nums = unit_nums[0]

    if not isinstance(din, list):
        din = [din]

    out = {}
    time = None
    with tables.open_file(h5_file, "r") as hf5:
        if din[0] is None and len(din) == 1:
            dins = [x._v_name for x in hf5.list_nodes("/spike_trains")]
        else:
            dins = ["dig_in_%i" % x for x in din if x is not None]

        for dig_str in dins:
            st = hf5.root.spike_trains[dig_str]
            tmp_time = st["array_time"][:]
            if time is None:
                time = tmp_time
            elif not np.array_equal(time, tmp_time):
                raise ValueError("Misaligned time vectors encountered")

            spike_array = st["spike_array"][:, unit_nums, :]
            out[dig_str] = spike_array

    if (
            isinstance(trials, int)
            or isinstance(trials, np.int32)
            or isinstance(trials, np.int64)
    ):
        for k in out.keys():
            out[k] = out[k][:trials]

    elif trials is not None:
        for k in out.keys():
            out[k] = out[k][trials]

    if len(out) == 1:
        out = out.popitem()[1]

    return time, out


def get_raw_digital_signal(rec_dir, dig_type, channel, h5_file=None):
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    with tables.open_file(h5_file, "r") as hf5:
        if (
                "/digital_%s" % dig_type in hf5
                and "/digital_%s/dig_%s_%i" % (dig_type, dig_type, channel) in hf5
        ):
            out = hf5.root["digital_%s" % dig_type]["dig_%s_%i" % (dig_type, channel)][:]
            return out
    return None


def get_raw_trace(h5_file=None, rec_dir=None, chan_idx=None, ):
    """
    Returns raw voltage trace for electrode from hdf5 store.
    """
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir, shell=True)

    with tables.open_file(h5_file, "r") as hf5:
        if "/raw" in hf5 and f"/raw/electrode{chan_idx}" in hf5:
            out = hf5.root.raw[f"electrode{chan_idx}"][:]
            return out
        else:
            return None


def _check_electrode_data_exists(h5_file=None, rec_dir=None, chan_idx=None, ):
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir, shell=True)

    with tables.open_file(h5_file, "r") as hf5:
        if "/raw" in hf5 and f"/raw/electrode{chan_idx}" in hf5:
            return True
        else:
            return False


def get_referenced_trace(rec_dir, electrode, h5_file=None):
    """Returns referenced voltage trace for electrode from hdf5 store
    If /referenced is not in hdf5, return None

    Parameters
    ----------
    rec_dir : str, recording directory
    electrode : int

    Returns
    -------
    np.array of the scaled referenced voltage trace or None if referenced trace
    could not be obtained
    """
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    with tables.open_file(h5_file, "r") as hf5:
        if "/referenced" in hf5 and "/referenced/electrode%i" % electrode in hf5:
            out = hf5.root.referenced["electrode%i" % electrode][:] * 1.0
        else:
            out = None

    return out


def get_raw_unit_waveforms(
        rec_dir,
        unit_name,
        electrode_mapping=None,
        clustering_params=None,
        shell=True,
        required_descrip=None,
        h5_file=None,
):
    """
    Returns the waveforms of a single unit extracting them directly from the
    raw data files.

    Parameters
    ----------
    rec_dir : str
    unit_name : str or int
    electrode_mapping : pd.DataFrame (optional)
        if raw data is not in hf5 this is used to get the raw data from the
        original dat files
        if not provided program will try to grab this from the h5 file
    clustering_params : dict (optional)
        requires fields 'spike_snapshot' and 'bandpass_params'
        if not provided these will be queried from the json file written in the
        rec_dir or from the defaults if no params are in rec_dir
    shell : bool (optional)
        True to CLI (default in ssh). False for GUI. Only relevant if multiple
        h5 files in rec_dir
    required_descrip : tuple (optional)
        required unit description. Returns None instead of raw trace if unit
        does not match required_descrip

    Returns
    -------
    raw_trace, unit_descriptor, sampling_rate
    np.array, tuple, float
    """
    if isinstance(unit_name, int):
        unit_num = unit_name
        unit_name = "unit%03i" % unit_num
    else:
        unit_num = parse_unit_number(unit_name)

    if electrode_mapping is None:
        electrode_mapping = get_electrode_mapping(rec_dir)

    if clustering_params is None:
        clustering_params = paramio.load_params("clustering_params", rec_dir)

    snapshot = clustering_params["spike_snapshot"]
    snapshot = [snapshot["Time before spike (ms)"], snapshot["Time after spike (ms)"]]
    bandpass = clustering_params["bandpass_params"]
    bandpass = [bandpass["Lower freq cutoff"], bandpass["Upper freq cutoff"]]
    fs = clustering_params["sampling_rate"]
    if fs is None:
        raise ValueError("clustering_params.json does not exist")

    # Get spike times for unit
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir, shell=shell)

    with tables.open_file(h5_file, "r") as hf5:
        spike_times = hf5.root.sorted_units[unit_name]["times"][:]
        descriptor = hf5.root.unit_descriptor[unit_num]

    if required_descrip is not None:
        if descriptor != required_descrip:
            return None, descriptor, fs

    print("Getting raw waveforms for %s %s" % (os.path.basename(rec_dir), unit_name))
    electrode = descriptor["electrode_number"]
    raw_el = get_raw_trace(rec_dir, electrode, electrode_mapping)
    if electrode_mapping is None and raw_el is None:
        raise FileNotFoundError(
            "Raw data not found in h5 file and" " electrode_mapping not found"
        )

    if raw_el is None:
        raise FileNotFoundError("Raw data not found")

    slices_dj, new_fs = clust.get_waveforms(
        raw_el, spike_times, sampling_rate=fs, snapshot=snapshot, bandpass=bandpass
    )
    return slices_dj, descriptor, new_fs


def write_time_vector_to_h5(h5_file, electrode, fs):
    """Writes time vector the same size as the electrode array"""
    with tables.open_file(h5_file, "r+") as hf5:

        if "/raw/electrode%i" % electrode in hf5:
            arr = hf5.root.raw["electrode%i" % electrode][:]
            time = np.arange(0, arr.shape[0])
            hf5.root.time.time_vector.append(time)
            return True
        else:
            return False


def write_spike2_array_to_h5(h5_file, electrode, waves, fs=None):
    if not Path(h5_file).exists():
        h5_file = get_h5_filename(h5_file)

    if len(waves) == 0:
        return False

    println("Writing electrode%i to %s..." % (electrode, h5_file))
    with tables.open_file(h5_file, "r+") as hf5:
        if "/raw" in hf5 and "/raw/electrode%i" % electrode in hf5:
            node = hf5.root.raw["electrode%i" % electrode]
            node._v_attrs["has_data"] = True
            if fs is not None:
                node._v_attrs["sampling_rate"] = fs
            node.append(waves)
            return True


def get_unit_waveforms(file_dir, unit, required_descrip=None, h5_file=None):
    if isinstance(unit, int):
        un = "unit%03i" % unit
    else:
        un = unit
        unit = parse_unit_number(un)

    if h5_file is None:
        h5_file = get_h5_filename(file_dir)

    clustering_params = paramio.load_params("clustering_params", file_dir)
    fs = clustering_params["sampling_rate"]
    with tables.open_file(h5_file, "r") as hf5:
        waveforms = hf5.root.sorted_units[un].waveforms[:]
        descriptor = hf5.root.unit_descriptor[unit]

    if required_descrip is not None:
        if descriptor != required_descrip:
            return None, descriptor, fs

    return waveforms, descriptor, fs * 10


def get_unit_spike_times(file_dir, unit, required_descrip=None, h5_file=None):
    if isinstance(unit, int):
        un = "unit%03i" % unit
    else:
        un = unit
        unit = parse_unit_number(un)

    if h5_file is None:
        h5_file = get_h5_filename(file_dir)

    clustering_params = paramio.load_params("clustering_params", file_dir)
    fs = clustering_params["sampling_rate"]
    with tables.open_file(h5_file, "r") as hf5:
        times = hf5.root.sorted_units[un].times[:]
        descriptor = hf5.root.unit_descriptor[unit]

    if required_descrip is not None:
        if descriptor != required_descrip:
            return None, descriptor, fs

    return times, descriptor, fs * 10


def get_unit_as_cluster(file_dirs, unit, rec_key=None):
    if isinstance(unit, int):
        un = "unit%03i" % unit
    else:
        un = unit
        unit = parse_unit_number(unit)

    if isinstance(file_dirs, str):
        file_dirs = [file_dirs]

    if rec_key is None:
        rec_key = {k: v for k, v in enumerate(file_dirs)}

    waves = []
    times = []
    spike_map = []
    fs = dict.fromkeys(rec_key.keys())
    offsets = dict.fromkeys(rec_key.keys())
    offset = 0
    for k in sorted(rec_key.keys()):
        v = rec_key[k]
        tmp_waves, descriptor, tmp_fs = get_unit_waveforms(v, unit)
        tmp_times, _, _ = get_unit_spike_times(v, unit)
        waves.append(tmp_waves)
        times.append(tmp_times)
        tmp_map = np.ones(tmp_times.shape) * k
        spike_map.append(tmp_map)
        fs[k] = tmp_fs
        em = get_electrode_mapping(v)
        el = descriptor["electrode_number"]
        offsets[k] = offset
        offset += 3 * tmp_fs + em.query("Electrode==@el")["cutoff_time"].values[0]

    spike_map = np.hstack(spike_map)
    times = np.hstack(times)
    waves = np.vstack(waves)
    clusters = {
        "Cluster_Name": un,
        "solution_num": 0,
        "cluster_num": unit,
        "cluster_id": unit,
        "spike_waveforms": waves,
        "spike_times": times,
        "spike_map": spike_map,
        "rec_key": rec_key,
        "fs": fs,
        "offsets": offsets,
        "manipulations": "",
    }

    return clusters


def get_electrode_mapping(rec_dir, h5_file=None):
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)
    with tables.open_file(h5_file, "r") as hf5:
        if "/electrode_map" not in hf5:
            return None
        table = hf5.root.electrode_map[:]
        el_map = read_table_into_DataFrame(table)
    return el_map


def get_digital_mapping(rec_dir, dig_type, h5_file=None):
    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    with tables.open_file(h5_file, "r") as hf5:
        if ("/digital_%sput_map" % dig_type) not in hf5:
            return None

        table = hf5.root["digital_%sput_map" % dig_type][:]
        dig_map = read_table_into_DataFrame(table)

    return dig_map


def get_node_list(h5_file):
    with tables.open_file(h5_file, "r") as hf5:

        def list_nodes(node):
            out = [node._v_name]
            n = node._v_name
            if not isinstance(node, tables.group.Group):
                return out

            nc = node._v_nchildren
            if nc > 0:
                nodes = hf5.list_nodes(node)
                for child in nodes:
                    out.extend(["%s.%s" % (n, x) for x in list_nodes(child)])

            return out

        nodes = hf5.list_nodes("/")
        out = []
        for node in nodes:
            out.extend(list_nodes(node))

        return out


def get_recording_filetype(file_dir):
    """
    Check Intan recording directory to determine type of recording and thus
    extraction method to use. Asks user to confirm, and manually correct if
    incorrect

    Parameters
    ----------
    file_dir : str, recording directory to check

    Returns
    -------
    str : file_type of recording
    """
    file_list = os.listdir(file_dir)
    file_type = None
    for k, v in SUPP_REC_TYPES.items():
        regex = re.compile(v)
        if any([True for x in file_list if regex.match(x) is not None]):
            file_type = k

    if file_type is None:
        msg = "\n   ".join(
            [
                "unsupported recording type. Supported types are:",
                *list(SUPP_REC_TYPES.keys()),
            ]
        )
    else:
        msg = '"' + file_type + '"'

    return file_type


def write_electrode_map_to_h5(h5_file, electrode_map):
    """Writes electrode mapping DataFrame to table in hdf5 store"""
    print("Writing electrode_map to %s..." % h5_file)
    with tables.open_file(h5_file, "r+") as hf5:
        if "/electrode_map" in hf5:
            hf5.remove_node("/", "electrode_map")

        table = hf5.create_table(
            "/", "electrode_map", particles.electrode_map_particle, "Electrode Map"
        )
        new_row = table.row
        for i, row in electrode_map.iterrows():
            for k, v in row.items():
                if k not in table.colnames:
                    continue
                if pd.isna(v):
                    if type(new_row[k]) == str:
                        v = "None"
                    else:
                        v = -1
                new_row[k] = v
            new_row.append()
        hf5.flush()


def write_digital_map_to_h5(h5_file, digital_map, dig_type):
    """Write digital input/output mapping DataFrame to table in hdf5 store"""
    dig_str = "digital_%sput_map" % dig_type
    print("Writing %s to %s..." % (dig_str, h5_file))
    with tables.open_file(h5_file, "r+") as hf5:
        if ("/" + dig_str) in hf5:
            hf5.remove_node("/%s" % dig_str)

        table = hf5.create_table(
            "/",
            dig_str,
            particles.digital_mapping_particle,
            "Digital %sput Map" % dig_type,
        )
        new_row = table.row
        for i, row in digital_map.iterrows():
            for k, v in row.items():
                new_row[k] = row[k]

            new_row.append()

        hf5.flush()


def read_table_into_DataFrame(table):
    df = pd.DataFrame.from_records(table)
    dt = df.dtypes
    idx = np.where(dt == "object")[0]
    for i in idx:
        k = dt.keys()[i]
        df[k] = df[k].apply(lambda x: x.decode("utf-8"))

    return df


def read_unit_description(unit_description):
    try:
        rsu = bool(unit_description["regular_spiking"])
        fsu = bool(unit_description["fast_spiking"])
    except ValueError:
        raise ValueError("Not a proper unit description")

    if rsu and fsu:
        return "Mislabelled"
    elif rsu:
        return "Regular-spiking"
    elif fsu:
        return "Fast-spiking"
    else:
        return "Unlabelled"


def add_new_unit(
        rec_dir, electrode, waves, times, single_unit, multi_unit, h5_file=None
):
    """
    Adds new sorted unit to h5_file and returns the new unit name
    Creates new row for unit description and add waveforms and times arrays

    Parameters
    ----------
    rec_dir : str
    electrode : int
    waves : np.array
    times : np.array
    single_unit : bool or int
    multi_unit : bool or int
    h5_file : str (optional)

    Returns
    -------
    str : unit_name
    """

    if h5_file is None:
        h5_file = get_h5_filename(rec_dir)

    unit_name = get_next_unit_name(rec_dir)

    with tables.open_file(h5_file, "r+") as hf5:
        if "/sorted_units" not in hf5:
            print("Creating sorted_units group")
            hf5.create_group("/", "sorted_units")

        if "/unit_descriptor" not in hf5:
            print("Creating unit_descriptor table")
            hf5.create_table("/", "unit_descriptor", description=particles.unit_descriptor)

        table = hf5.root.unit_descriptor
        unit_descrip = table.row

        for x in unit_descrip:
            print(x)

        sys.stdout.flush()

        unit_descrip["electrode_number"] = int(electrode)
        unit_descrip["single_unit"] = int(single_unit)
        try:
            unit_descrip["multi_unit"] = int(multi_unit)
        except KeyError:
            # add the multi_unit column if it doesn't exist
            table.row.append()
            unit_descrip["multi_unit"] = int(multi_unit)

        hf5.create_group("/sorted_units", unit_name, title=unit_name)
        waveforms = hf5.create_array("/sorted_units/%s" % unit_name, "waveforms", waves)
        times = hf5.create_array("/sorted_units/%s" % unit_name, "times", times)
        unit_descrip.append()
        table.flush()
        hf5.flush()

    return unit_name


def edit_unit_descriptor(
        file_dir, unit_num, descriptor_key, descriptor_val, h5_file=None
):
    """
    use this to edit unit table, i.e. if you made a mistake labeling a neuron in spike sorting
    unit_num takes integers, corresponds to unit_num in get_unit_table()
    descriptor_key takes string, can be "single_unit", "regular_spiking", or "fast_spiking"
    descriptor_val takes boolean, can be True or False
    """
    if isinstance(unit_num, str):
        unit_num = parse_unit_number(unit_num)

    print("\n----------\n editing unit %i descriptor\n----------\n" % unit_num)
    if h5_file is None:
        h5_file = get_h5_filename(file_dir)

    unit_names = get_unit_names(file_dir)
    unit_numbers = [parse_unit_number(i) for i in unit_names]
    if unit_num not in unit_numbers:
        print("Unit %i not found in data. Cannot edit descriptor " % unit_num)
        return False

    with tables.open_file(h5_file, "r+") as hf5:
        unit_descriptor = hf5.root.unit_descriptor[unit_num]
        unit_descriptor[descriptor_key] = descriptor_val
        hf5.root.unit_descriptor[[unit_num]] = unit_descriptor
        hf5.flush()

    return

def check_h5_data(filename):
    with (tables.open_file(filename, "r") as hf5):
        nodes = hf5.root._f_list_nodes()
        node_data = {}
        for node in nodes:
            if node._v_name in DATA_GROUPS:
                if node._v_nchildren > 0:
                    for child in node._f_list_nodes():
                        if hasattr(child, "nrows"):
                            if child.nrows > 0:
                                node_data[child._v_name] = child.shape
    return node_data


def create_empty_data_h5(filename, overwrite=False, shell=False):
    """
    Create empty h5 store for data with approriate data groups

    Parameters
    ----------
    filename : str, absolute path to h5 file for recording
    """

    if "SHH_CONNECTION" in os.environ:
        shell = True

    if not hasattr(filename, "endswith"):
        filename = str(filename)
    if not filename.endswith(".h5") and not filename.endswith(".hdf5"):
        filename += ".h5"

    basename = os.path.splitext(os.path.basename(filename))[0]

    if os.path.isfile(filename):
        node_data = check_h5_data(filename)
        if overwrite or node_data == {}:
            q = 1
        else:
            # gather h5 information
            q = userio.ask_user(
                f"{filename} already exists. Overwrite? \n"
                f"{pprint.pformat(node_data)}",
                choices=["No", "Yes"],
                shell=shell,
            )

        if q == 0:
            return None
        else:
            println("Deleting existing h5 file...")
            os.remove(filename)
            print("Done!")

    print("Creating empty HDF5 store with raw data groups")
    println("Writing %s.h5 ..." % basename)
    with tables.open_file(filename, "w", title=basename) as hf5:
        for grp in DATA_GROUPS:
            hf5.create_group("/", grp)
        hf5.flush()

    print("Done!\n")
    return filename

def create_hdf_arrays(
        file_name: str | Path,
        rec_info: dict,
        electrode_mapping: pd.DataFrame = None,
        lfp_mapping: pd.DataFrame = None,
        event_mapping: pd.DataFrame = None,
) -> None:
    file_name = Path(file_name)
    if not file_name.suffix == ".h5":
        file_name = file_name.with_suffix(".h5")

    println("Creating empty arrays in hdf5 store for raw data...")
    atom = tables.IntAtom()
    f_atom = tables.Float64Atom()

    with tables.open_file(str(file_name), "r+") as hf5:
        # Create array for raw time vector
        hf5.create_earray("/time", "time_vector", f_atom, (0,))

        if electrode_mapping is not None:  # only ones to sort
            if not electrode_mapping.empty:
                for idx, row in electrode_mapping.iterrows():
                    hf5.create_earray(
                        "/raw", f"electrode{row['electrode']}", f_atom, (0,)
                    )
                    hf5.root.raw._v_attrs["sampling_rate"] = row["sampling_rate"]
                    hf5.root.raw._v_attrs["units"] = row["units"]
                hf5.root.raw._v_attrs["num_electrodes"] = len(electrode_mapping)

        if lfp_mapping is not None:
            if not lfp_mapping.empty:
                for idx, row in lfp_mapping.iterrows():
                    hf5.create_earray(
                        "/raw_lfp", f"lfp{row['electrode']}", f_atom, (0,)
                    )
                    # attach the sampling rate to the lfp group
                    hf5.root.raw_lfp._v_attrs["sampling_rate"] = row["sampling_rate"]
                hf5.root.raw_lfp._v_attrs["num_electrodes"] = len(lfp_mapping)

        if rec_info.get("dig_in") is not None:
            for x in rec_info["dig_in"]:
                hf5.create_earray("/digital_in", "dig_in_%i" % x, atom, (0,))

        # Create arrays for digital outputs (if any exist)
        if rec_info.get("dig_out") is not None:
            for x in rec_info["dig_out"]:
                hf5.create_earray("/digital_out", "dig_out_%i" % x, atom, (0,))

    print("Done!")


def write_array_to_hdf5(hf5, loc, name, arr):
    try:
        hf5.create_array(loc, name, arr)
        hf5.flush()
    except tables.exceptions.NodeError:
        hf5.remove_node(loc, name)
        hf5.create_array(loc, name, arr)
        hf5.flush()


def delete_unit(file_dir, unit_num, h5_file=None):
    """Delete a sorted unit and re-label all following units.

    Parameters
    ----------
    file_dir : str, full path to recording directory
    unit_num : int, number of unit to delete
    """
    if isinstance(unit_num, str):
        unit_num = parse_unit_number(unit_num)

    print("\n----------\nDeleting unit %i from dataset\n----------\n" % unit_num)
    if h5_file is None:
        h5_file = get_h5_filename(file_dir)

    unit_names = get_unit_names(file_dir)
    unit_numbers = [parse_unit_number(i) for i in unit_names]
    if unit_num not in unit_numbers:
        print("Unit %i not found in data. Nothing being deleted" % unit_num)
        return False

    metrics_dir = os.path.join(file_dir, "sorted_unit_metrics")
    plot_dir = os.path.join(file_dir, "unit_waveforms_plots")

    unit_name = "unit%03d" % unit_num
    change_units = [x for x in unit_numbers if x > unit_num]
    new_units = [x - 1 for x in change_units]
    new_names = ["unit%03d" % x for x in new_units]
    old_names = ["unit%03d" % x for x in change_units]
    old_prefix = ["Unit%i" % x for x in change_units]
    new_prefix = ["Unit%i" % x for x in new_units]

    # Remove metrics
    if os.path.exists(os.path.join(metrics_dir, unit_name)):
        shutil.rmtree(os.path.join(metrics_dir, unit_name))

    # remove unit from hdf5 store
    with tables.open_file(h5_file, "r+") as hf5:
        hf5.remove_node("/sorted_units", name=unit_name, recursive=True)
        table = hf5.root.unit_descriptor
        table.remove_row(unit_num)
        # rename rest of units in hdf5 and metrics folders
        print("Renaming following units...")
        for x, y in zip(old_names, new_names):
            print("Renaming %s to %s" % (x, y))
            hf5.rename_node("/sorted_units", newname=y, name=x)
            os.rename(os.path.join(metrics_dir, x), os.path.join(metrics_dir, y))

        hf5.flush()

    # delete and rename plot files
    if os.path.exists(plot_dir):
        swap_files = [
            ("Unit%i.png" % x, "Unit%i.png" % y)
            for x, y in zip(change_units, new_units)
        ]
        swap_files2 = [
            ("Unit%i_mean_sd.png" % x, "Unit%i_mean_sd.png" % y)
            for x, y in zip(change_units, new_units)
        ]
        swap_files.extend(swap_files2)
        del_plots = ["Unit%i.png" % unit_num, "Unit%i_mean_sd.png" % unit_num]
        print("Correcting names of plots and metrics...")
        for x in del_plots:
            if os.path.exists(os.path.join(plot_dir, x)):
                os.remove(os.path.join(plot_dir, x))

        for x in swap_files:
            if os.path.exists(os.path.join(plot_dir, x[0])):
                os.rename(os.path.join(plot_dir, x[0]), os.path.join(plot_dir, x[1]))

    # compress_and_repack(h5_file)
    print("Finished deleting unit\n----------")
    return True


def parse_unit_number(unit_name):
    """number of unit extracted from unit_name

    Parameters
    ----------
    unit_name : str, unit###

    Returns
    -------
    int
    """
    pattern = "unit(\d*)"
    parser = re.compile(pattern)
    out = int(parser.match(unit_name)[1])
    return out


def fix_unit_numbering(file_dir):
    """checks all units in an h5 file and makes sure all numbers are
    continuous. Corrects if not.

    Parameters
    ----------
    file_dir : str, path to recording dir that contains h5 file
    """
    print("\n----------\nCorrecting unit names in dataset\n----------\n")

    unit_table = get_unit_table(file_dir)
    if all(unit_table.unit_num == unit_table.index):
        print("No Unit Names Changed. All good!")
        return True

    change_map = {x: y for x, y in zip(unit_table.unit_num, unit_table.index)}
    h5_file = get_h5_filename(file_dir)

    metrics_dir = os.path.join(file_dir, "sorted_unit_metrics")
    plot_dir = os.path.join(file_dir, "unit_waveforms_plots")

    with tables.open_file(h5_file, "a") as hf5:
        for x, y in change_map.items():
            u1 = "unit%03d" % x
            u2 = "unit%03d" % y
            if x == y:
                continue

            print(f"Renaming {u1} to {u2}")
            hf5.rename_node("/sorted_units", newname=u2, name=u1)
            m1 = os.path.join(metrics_dir, u1)
            m2 = os.path.join(metrics_dir, u2)
            if os.path.isdir(m1):
                os.rename(m1, m2)

            p1 = os.path.join(plot_dir, f"Unit{x}.png")
            p2 = os.path.join(plot_dir, f"Unit{y}.png")
            q1 = os.path.join(plot_dir, f"Unit{x}_mean_sd.png")
            q2 = os.path.join(plot_dir, f"Unit{y}_mean_sd.png")
            if os.path.isfile(p1):
                os.rename(p1, p2)
            if os.path.isfile(q1):
                os.rename(q1, q2)

        hf5.flush()

    print("Finished correcting unit names\n----------")
    return True


def common_avg_reference(h5_file, electrodes, group_num):
    """Computes and subtracts the common average for a group of electrodes

    Parameters
    ----------
    h5_file : str, path to .h5 file with the raw data
    electrodes : list of int, electrodes to average
    group_num : int, number of common average group (for  storing common
                     average in hdf5 store)
    """
    if not os.path.isfile(h5_file):
        raise FileNotFoundError("%s was not found." % h5_file)

    print(
        "Common Average Referencing Electrodes:\n"
        + ", ".join([str(x) for x in electrodes.copy()])
    )

    with tables.open_file(h5_file, "r+") as hf5:
        raw = hf5.root.raw
        samples = np.array([raw["electrode%i" % x][:].shape[0] for x in electrodes])
        min_samples = np.min(samples)
        if any(samples != min_samples):
            print(
                "Some raw voltage traces are different lengths.\n"
                "This could be a sign that recording was cutoff early.\n"
                "Truncating to the length of the shortest trace for analysis"
                "\n    Min Samples: %i\n    Max Samples: %i"
                % (min_samples, np.max(samples))
            )

        # Calculate common average
        println("Computing common average...")
        common_avg = np.zeros((1, min_samples))[0]

        for x in electrodes:
            common_avg += raw["electrode%i" % x][:min_samples]

        common_avg /= float(len(electrodes))
        print("Done!")

        # Store common average
        Atom = tables.Float64Atom()
        println("Storing common average signal...")
        if "/common_average" not in hf5:
            hf5.create_group(
                "/", "common_average", "Common average electrodes and signals"
            )

        if "/common_average/electrodes_group%i" % group_num in hf5:
            hf5.remove_node("/common_average/electrodes_group%i" % group_num)

        if "/common_average/common_average_group%i" % group_num in hf5:
            hf5.remove_node("/common_average/common_average_group%i" % group_num)

        hf5.create_array(
            "/common_average", "electrodes_group%i" % group_num, np.array(electrodes)
        )
        hf5.create_earray(
            "/common_average", "common_average_group%i" % group_num, obj=common_avg
        )
        hf5.flush()
        print("Done!")

        # Replace raw data with referenced data
        println("Storing referenced signals...")
        for x in electrodes:
            referenced_data = raw["electrode%i" % x][:min_samples] - common_avg
            hf5.remove_node("/raw/electrode%i" % x)

            if "/referenced" not in hf5:
                hf5.create_group("/", "referenced", "Common average referenced signals")

            if "/referenced/electrode%i" % x in hf5:
                hf5.remove_node("/referenced/electrode%i" % x)

            hf5.create_earray("/referenced", "electrode%i" % x, obj=referenced_data)
            hf5.flush()

        print("Done!")


def compress_and_repack(h5_file, new_file=None):
    """
    Compress and repack the h5 file with ptrepack either to same name or new name

    Parameters
    ----------
    h5_file : str, path to h5 file
    new_file : str (optional), new path for h5_file

    Returns
    -------
    str, new path to h5 file
    """
    if new_file is None or new_file == h5_file:
        new_file = os.path.join(os.path.dirname(h5_file), "tmp.h5")
        tmp = True
    else:
        tmp = False

    print("Repacking %s as %s..." % (h5_file, new_file))
    subprocess.call(
        [
            "ptrepack",
            "--chunkshape",
            "auto",
            "--propindexes",
            "--complevel",
            "9",
            "--complib",
            "blosc",
            h5_file,
            new_file,
        ]
    )

    # Remove old  h5 file
    print("Removing old h5 file: %s" % h5_file)
    os.remove(h5_file)

    # If used a temporary rename to old file name
    if tmp:
        print("Renaming temporary file to %s" % h5_file)
        subprocess.call(["mv", new_file, h5_file])
        new_file = h5_file

    return new_file


def cleanup_clustering(file_dir, h5_file=None):
    """
    Consolidate memory monitor files from clustering, remove raw and
    referenced data from hdf5 and repack

    Parameters
    ----------
    file_dir : str, path to recording directory

    Returns
    -------
    str, path to new hdf5 file
    """
    # Grab h5 filename
    if h5_file is None:
        h5_file = get_h5_filename(file_dir)

    # If raw and/or referenced data is still in h5
    # Remove raw/referenced data from hf5
    # Repack h5 as *_repacked.h5
    # Create sorted_units groups in h5, if it doesn't exist
    changes = False
    with tables.open_file(h5_file, "r+") as hf5:
        if "/raw" in hf5:
            # println("Removing raw data from hdf5 store...")
            # hf5.remove_node("/raw", recursive=1)
            # changes = True
            ic(f"Leaving raw store... {h5_file}")

        if "/sorted_units" not in hf5:
            hf5.create_group("/", "sorted_units")
            changes = True

        if "/unit_descriptor" not in hf5:
            hf5.create_table(
                "/", "unit_descriptor", description=particles.unit_descriptor
            )
            changes = True

    # Repack if any big changes were made to h5 store
    if changes:
        if not hasattr(h5_file, "endswith"):  #  pathlib / os.path compatibility
            h5_file = str(h5_file)
        if h5_file.endswith("_repacked.h5"):
            new_fn = h5_file
            new_h5 = compress_and_repack(h5_file, new_fn)
        else:
            ic("Renaming..")
            new_fn = h5_file.replace(".h5", "_repacked.h5")
            new_h5 = compress_and_repack(h5_file)
        return new_h5
    else:

        return h5_file


def create_trial_data_table(h5_file, digital_map, fs, dig_type="in"):
    """
    Returns trial data: trial num, spk_io #, spk_io name, on times, off times

    Parameters
    ----------
    h5_file : str, full path to hdf5 store
    digital_map : pandas.DataFrame
        maps digital channel numbers to string names,
        has columns 'channel' and 'name'
    fs : float, sampling rate in Hz
    channels : list of int (optional)
        DIN or DOUT channel numbers to return data from. None (default)
        returns data for all channels
    dig_type : {'in', 'out'}, whether to return digital input or output data

    Returns
    -------
    pandas.DataFrame with columns:
        trial_num, channel, name, on_time, off_time, duration
    """
    if dig_type not in ["in", "out"]:
        raise ValueError("Invalid digital type given.")

    rec_dir = os.path.dirname(h5_file)
    trial_map = []
    print(
        "Generating trial list for digital %sputs: %s"
        % (dig_type, ", ".join([str(x) for x in digital_map["channel"].tolist()]))
    )
    for i, row in digital_map.iterrows():
        channel = row["channel"]
        exp_start_idx = 0
        exp_end_idx = 0
        dig_trace = get_raw_digital_signal(rec_dir, dig_type, row["channel"])
        if dig_trace is None:
            print(
                f"No signal trace found for digital {dig_type}put "
                "#{channel}. Skipping..."
            )
            continue

        if len(dig_trace) > exp_end_idx:
            exp_end_idx = len(dig_trace)

        dig_diff = np.diff(dig_trace)
        on_idx = np.where(dig_diff > 0)[0]
        off_idx = np.where(dig_diff < 0)[0]
        trial_map.extend(
            [
                (x, row["channel"], row["name"], x, y, x / fs, y / fs)
                for x, y in zip(on_idx, off_idx)
            ]
        )

    # Add one more row for experiment start and end time
    trial_map.extend(
        [
            (
                0,
                -1,
                "Experiment",
                exp_start_idx,
                exp_end_idx,
                exp_start_idx / fs,
                exp_end_idx / fs,
            )
        ]
    )

    # Make dataframe and assign trial numbers
    println("Constructing DataFrame...")
    trial_df = pd.DataFrame(
        trial_map,
        columns=[
            "idx",
            "channel",
            "name",
            "on_index",
            "off_index",
            "on_time",
            "off_time",
        ],
    )
    trial_df = trial_df.sort_values(by=["idx"]).reset_index(drop=True)
    trial_df = trial_df.reset_index(drop=False).rename(columns={"index": "trial_num"})
    trial_df = trial_df.drop(columns=["idx"])
    print("Done!")
    with tables.open_file(h5_file, "r+") as hf5:
        # Make hf5 group and table
        println("Writing data to h5 file...")
        if "/trial_info" not in hf5:
            group = hf5.create_group("/", "trial_info", "Trial Lists")

        if "/trial_info/digital_%s_trials" % dig_type in hf5:
            hf5.remove_node(
                "/trial_info", "digital_%s_trials" % dig_type, recursive=True
            )

        table = hf5.create_table(
            "/trial_info",
            "digital_%s_trials" % dig_type,
            particles.trial_info_particle,
            "Trial List  for Digital %sputs" % dig_type,
        )
        new_row = table.row
        for i, row in trial_df.iterrows():
            new_row["trial_num"] = row["trial_num"]
            new_row["name"] = row["name"]
            new_row["channel"] = row["channel"]
            new_row["on_index"] = row["on_index"]
            new_row["off_index"] = row["off_index"]
            new_row["on_time"] = row["on_time"]
            new_row["off_time"] = row["off_time"]

            new_row.append()

        # make one more row for experiment info

        hf5.flush()
        print("Done!")

    return trial_df


def read_trial_data_table(h5_file, dig_type="in", channels=None):
    """Opens the h5 file and returns the digital_in or digital_out trial info
    as a pandas DataFrame. Can specify specific digital channels if desired.

    Parameters
    ----------
    h5_file : str, path to h5_file
    dig_type : {'in' (default), 'out'}
        which type of digital signal to return table for
    channels : list[int] (optional)
        which digital channels to return info for.
        None (default) returns data for all channels

    Returns
    -------
    pandas.DataFrame
        with columns:
            - channel
            - name
            - trial_num
            - on_index
            - off_index
            - on_time
            - off_time
    """
    if not os.path.isfile(h5_file):
        raise FileNotFoundError("%s was not found" % h5_file)

    trial_node = "/trial_info/digital_%s_trials" % dig_type

    with tables.open_file(h5_file, "r") as hf5:
        if "/trial_info" not in hf5 or trial_node not in hf5:
            raise ValueError("trial_info table not found in hdf5 store")

        df = pd.DataFrame.from_records(hf5.get_node(trial_node)[:])
        df["name"] = df["name"].apply(lambda x: x.decode("utf-8"))

    if channels is not None:
        df = df[df["channel"].isin(channels)]

    return df


def read_in_amplifier_signal(hf5, file_dir, file_type, num_channels, el_map, em_map):
    """Read intan amplifier files into hf5 array.
    For electrode and emg signals.
    Supported recording types:
        - one file per signal type
        - one file per channel

    Parameters
    ----------
    hf5 : tables.file.File, hdf5 object to write data into
    file_dir : str, path to recording directory
    file_type : str
        type of recording files to read in. Currently supported: 'one file per
        signal type' and 'one file per channel'
    num_channels: int
        number of amplifier channels from info.rhd or
        blechby.rawIO.read_recording_info
    el_map, em_map : pandas.DataFrames
        dataframe mapping electrode or emg number to port and channel numer.
        Must have columns Port and Channel and either Electrode (el_map) or EMG
        (em_map)
    """
    exec_str = "hf5.root.%s.%s%i.append(data[:])"

    # Read in electrode data
    for idx, row in el_map.iterrows():
        port = row["Port"]
        channel = row["Channel"]
        electrode = row["Electrode"]

        if file_type == "one file per signal type":
            data = None
        elif file_type == "one file per channel":
            file_name = os.path.join(file_dir, "amp-%s-%03d.dat" % (port, channel))
            println("Reading data from %s..." % os.path.basename(file_name))
            data = 1
            print("Done!")

        tmp_str = exec_str % ("raw", "electrode", electrode)
        println(
            "Writing data from port %s channel %i to electrode%i..."
            % (port, channel, electrode)
        )
        exec(tmp_str)
        print("Done!")
        hf5.flush()

    # Read in emg data if it exists
    if not em_map.empty:
        for idx, row in em_map.iterrows():
            port = row["Port"]
            channel = row["Channel"]
            emg = row["EMG"]

            if file_type == "one file per signal type":
                data = None
            elif file_type == "one file per channel":
                file_name = os.path.join(file_dir, "amp-%s-%03d.dat" % (port, channel))
                println("Reading data from %s..." % os.path.basename(file_name))
                data = None
                print("Done!")

            tmp_str = exec_str % ("raw_emg", "emg", emg)
            println(
                "Writing data from port %s channel %i to emg%i..."
                % (port, channel, emg)
            )
            exec(tmp_str)
            print("Done!")
            hf5.flush()
