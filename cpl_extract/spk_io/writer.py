from pathlib import Path

from cpl_extract.spk_io import prompt
import pandas as pd
import json
import os


def write_dict_to_json(dat, save_file):
    """writes a dict to a json file

    Parameters
    ----------
    dat : dict
    save_file : str | Path
    """
    for k, v in dat.items():  # Path() objects are not serializable, so convert to str
        if isinstance(v, Path):  # :(
            dat[k] = str(v)
    with open(save_file, "w") as f:
        json.dump(dat, f, indent=4)


def write_params_to_json(param_name, rec_dir, params, overwrite=False):
    """Writes params into a json file placed in the analysis_params folder in
    rec_dir with the name param_name.json

    Parameters
    ----------
    param_name : str, name of parameter file
    rec_dir : str, recording directory
    params : dict, paramters

    Args:
        overwrite:
    """
    if not param_name.endswith(".json"):
        param_name += ".json"

    p_dir = Path(rec_dir) / "analysis_params"
    p_dir.mkdir(parents=True, exist_ok=True)
    save_file = p_dir / param_name

    # Path() objects are not serializable, so convert to str
    for k, v in params.items():
        if isinstance(v, Path):
            params[k] = str(v)

    if not save_file.is_file():
        print(f"Writing {param_name} to {save_file}")
        write_dict_to_json(params, save_file)
        return True
    else:
        if overwrite:
            q = prompt.ask_user(f"**{param_name}** already exists, do you want to overwrite it?",)
            if q == 1:
                write_dict_to_json(params, save_file)
                return True
            else:
                print(f"Skipping {param_name}")
                return True
        else:
            return False

def read_dict_from_json(save_file):
    """reads dict from json file

    Parameters
    ----------
    save_file : str
    """
    with open(save_file, "r") as f:
        out = json.load(f)

    return out


def write_pandas_to_table(df, save_file, overwrite=False, shell=True):
    if os.path.isfile(save_file) and not overwrite:
        q = prompt.ask_user(
            "File already exists: %s\nDo you want to overwrite it?" % save_file,
            shell=shell,
        )
        if q == 0:
            return

    df.to_csv(save_file, sep="\t")


def read_pandas_from_table(fn):
    df = pd.read_csv(fn, sep="\t", index_col=0)
    return df
