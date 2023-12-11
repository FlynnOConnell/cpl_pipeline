import logging
import os
import time
import json
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
from cpl_extract.spk_io import printer as pt, prompt

SCRIPT_DIR = os.path.dirname(__file__)
PARAM_DIR = os.path.join(SCRIPT_DIR, "defaults")
PARAM_NAMES = [
    "CAR_params",
    "pal_id_params",
    "data_cutoff_params",
    "clustering_params",
    "bandpass_params",
    "spike_snapshot",
    "psth_params",
]


def Timer(heading):
    def real_timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            print("")
            print("----------\n%s\n----------" % heading)
            result = func(*args, **kwargs)
            print("Done! Elapsed Time: %1.2f" % (time.time() - start))
            return result

        return wrapper

    return real_timer


def parse_amplifier_files(file_dir):
    """
    parses the filenames of amp-*-*.dat files in file_dir and returns port and
    channel numbers
    for 'one file per channel' recordings

    deprecated: get ports and channels from rawIO.read_recording_info instead
    """
    file_list = os.listdir(file_dir)
    ports = []
    channels = []
    for f in file_list:
        if f.startswith("amp"):
            tmp = f.replace(".dat", "").split("-")
            if tmp[1] in ports:
                idx = ports.index(tmp[1])
                channels[idx].append(int(tmp[2]))
            else:
                ports.append(tmp[1])
                channels.append([int(tmp[2])])
    for c in channels:
        c.sort()
    return ports, channels


def parse_board_files(file_dir):
    """
    parses board-*-*.dat files and returns lists of DIN and DOUT channels
    for 'one file per channel' type recordings

    deprecated: get DIN and DOUT from rawIO.read_recording_info instead
    """
    file_list = os.listdir(file_dir)
    DIN = []
    DOUT = []
    for f in file_list:
        if f.startswith("board"):
            tmp = f.replace(".dat", "").split("-")
            if tmp[1] == "DIN":
                DIN.append(int(tmp[2]))
            elif tmp[1] == "DOUT":
                DOUT.append(int(tmp[2]))
    return DIN, DOUT


def get_ports(file_dir):
    """
    reads the data files in file_dir and returns a list of amplifier ports

    deprecated: get ports and channels from rawIO.read_recording_info instead
    """
    ports, ch = parse_amplifier_files(file_dir)
    return ports


def get_channels_on_port(file_dir, port):
    """
    reads files in file_dir to determine which amplifier channels are on port

    deprecated: get ports and channels from rawIO.read_recording_info instead
    """
    ports, ch = parse_amplifier_files(file_dir)
    try:
        idx = ports.index(port)
    except ValueError as error:
        raise ValueError(
            "Files for port %s not found in %s" % (port, file_dir)
        ) from error
    return ch[idx]


def get_sampling_rate(file_dir):
    """
    uses info.rhd in file_dir to get sampling rate of the data

    deprecated: get ports and channels from rawIO.read_recording_info instead
    """
    sampling_rate = np.fromfile(
        os.path.join(file_dir, "info.rhd"), dtype=np.dtype("float32")
    )
    sampling_rate = int(sampling_rate[2])
    return sampling_rate


def get_din_channels(file_dir):
    """
    returns a list of DIN channels read from filenames in file_dir

    deprecated: get ports and channels from rawIO.read_recording_info instead
    """
    DIN, DOUT = parse_board_files(file_dir)
    return DIN


def flatten_channels(ports, channels, emg_port=None, emg_channels=None):
    """takes all ports and all channels and makes a dataframe mapping ports and
    channels to electrode numbers from 0 to N
    excludes emg_channels if given

    Parameters
    ----------
    ports : list, list of port names, length equal to channels
    channels : list, list of channels number, corresponding to elements of ports
    emg_port : str (optional), prefix of port with EMG channel. Default is None
    emg_channels: list (optional), list of channels on emg_port used for emg

    Returns
    -------
    electrode_mapping : pandas.DataFrame,
                        3 columns: Electrode, Port and Channel
    emg_mapping : pandas.DataFrame,
                    3 columns: EMG, Port, and Channel

    Throws
    ------
    ValueError : if length of ports is not equal to length of channels
    """
    el_map = []
    em_map = []
    ports = ports.copy()
    channels = channels.copy()
    for idx, p in enumerate(zip(ports, channels)):
        if p[0] == emg_port and p[1] in emg_channels:
            em_map.append(p)
        else:
            el_map.append(p)

    map_df = pd.DataFrame(el_map, columns=["Port", "Channel"])
    map_df.sort_values(by=["Port", "Channel"], ascending=True, inplace=True)
    map_df.reset_index(drop=True, inplace=True)
    map_df = map_df.reset_index(drop=False).rename(columns={"index": "Electrode"})

    emg_df = pd.DataFrame(em_map, columns=["Port", "Channel"])
    emg_df.sort_values(by=["Port", "Channel"], ascending=True, inplace=True)
    emg_df.reset_index(drop=True, inplace=True)
    emg_df = emg_df.reset_index(drop=False).rename(columns={"index": "EMG"})
    return map_df, emg_df


def write_dict_to_json(dat, save_file):
    """writes a dict to a json file

    Parameters
    ----------
    dat : dict
    save_file : str
    """
    with open(save_file, "w") as f:
        json.dump(dat, f, indent=True)


def read_dict_from_json(save_file):
    """reads dict from json file

    Parameters
    ----------
    save_file : str
    """
    # TODO: add better error handling for issues that could arise tyring to read a json file
    try:
        with open(save_file, "r") as f:
            out = json.load(f)
        return out
    except (FileNotFoundError, JSONDecodeError) as error:
        if "logger" in globals():
            logger = globals()["logger"]
            logger.warn(error)
        else:
            print(error)
        return None


def load_params(param_name, rec_dir=None, default_keyword=None):
    param_name = param_name if param_name.endswith(".json") else param_name + ".json"
    default_file = Path(PARAM_DIR) / param_name
    rec_file = Path(rec_dir) / "analysis_params" / param_name if rec_dir else None

    if rec_file and rec_file.is_file():
        out = read_dict_from_json(str(rec_file))
        if out is None:
            out = read_dict_from_json(str(default_file))
            if out is None:
                print(
                    f"Unable to retrieve {param_name} from defaults or recording directory"
                )
                raise FileNotFoundError
        if default_keyword and default_keyword in out:
            out = out[default_keyword]
    elif default_file.is_file():
        print(
            f"{param_name} not found in recording directory. Pulling parameters from defaults"
        )
        out = read_dict_from_json(str(default_file))
        if out.get("multi") is True and default_keyword is None:
            print(f"Multiple defaults in {param_name} file, but no keyword provided")
            logger = logging.getLogger("cpl")
            logger.critical(
                f"Multiple defaults in {param_name} file, but no keyword provided"
            )
            raise ValueError(
                f"Multiple defaults in {param_name} file, but no keyword provided"
            )
        elif out and default_keyword:
            out = out.get(default_keyword)
            if out is None:
                print(f"No {param_name} found for keyword {default_keyword}")
                raise FileNotFoundError(
                    f"No {param_name} found for keyword {default_keyword}"
                )
        elif out is None:
            raise FileNotFoundError(f"{param_name} default file is empty")

    else:
        print(f"{param_name}.json not found in recording directory or in defaults")
        raise FileNotFoundError

    return out


def write_params_to_json(param_name, rec_dir, params):
    """Writes params into a json file placed in the analysis_params folder in
    rec_dir with the name param_name.json

    Parameters
    ----------
    param_name : str, name of parameter file
    rec_dir : str, recording directory
    params : dict, paramters
    """
    if not param_name.endswith(".json"):
        param_name += ".json"

    p_dir = os.path.join(rec_dir, "analysis_params")
    save_file = os.path.join(p_dir, param_name)
    print("Writing %s to %s" % (param_name, save_file))
    if not os.path.isdir(p_dir):
        os.mkdir(p_dir)

    write_dict_to_json(params, save_file)
