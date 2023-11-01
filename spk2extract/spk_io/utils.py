from __future__ import annotations

from pathlib import Path

import numpy as np


def is_h5py_compatible(data):
    if isinstance(data, np.ndarray):
        return True
    if isinstance(data, (int, float)):
        return True
    if isinstance(data, (list, tuple)):
        return all(isinstance(item, (int, float)) for item in data)
    if isinstance(data, str):
        return True
    return False


def get_spk2extract_path(path: Path | str) -> Path:
    """
    Get the path to the root spk2extract folder from a path to a file or folder in the spk2extract folder.
    """

    new_path = None
    path = Path(path)
    if "spk2extract" in str(path):
        for path_idx in range(len(path.parts) - 1, 0, -1):
            if path.parts[path_idx] == "spk2extract":
                new_path = Path(path.parts[0])
                for path_part in path.parts[1 : path_idx + 1]:
                    new_path = new_path.joinpath(path_part)
                break
    else:
        raise FileNotFoundError("The `spk2extract` folder was not found in path")
    return new_path


def read_npz_as_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as npz_data:
        return {k: npz_data[k] for k in npz_data.keys()}
