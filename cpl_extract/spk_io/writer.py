from pathlib import Path

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
        from cpl_extract.spk_io import userio
        q = userio.ask_user(
            "File already exists: %s\nDo you want to overwrite it?" % save_file,
            shell=shell,
        )
        if q == 0:
            return

    df.to_csv(save_file, sep="\t")


def read_pandas_from_table(fn):
    df = pd.read_csv(fn, sep="\t", index_col=0)
    return df
