import logging
import os
from pathlib import Path

from cpl_extract.spk_io.writer import read_dict_from_json, write_dict_to_json

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
    """
    Writes params into a json file placed in the analysis_params folder in
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
