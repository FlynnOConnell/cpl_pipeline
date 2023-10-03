from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np


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
                for path_part in path.parts[1:path_idx + 1]:
                    new_path = new_path.joinpath(path_part)
                break
    else:
        raise FileNotFoundError("The `spk2extract` folder was not found in path")
    return new_path
