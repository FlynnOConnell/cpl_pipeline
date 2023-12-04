# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from main import run
from spk_config import SortConfig

globs = globals()
if "DEFAULT_INI" not in globs:
    DEFAULT_INI = Path(__file__).parent / "default.ini"




def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = SortConfig(DEFAULT_INI)
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"
    print(args)

    return args, ops


def main():

    # params path, not manual
    path = Path().home() / "cpsort"
    params = SortConfig(path)
    run(params, parallel=True, overwrite=False)


if __name__ == "__main__":
    main()
