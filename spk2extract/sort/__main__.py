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


def add_args(parser: argparse.ArgumentParser):
    """
    Adds config options arguments to parser.
    """
    parser.add_argument(
        "--config", default=DEFAULT_INI, help="Path to configuration .ini file, including filename."
    )
    parser.add_argument("--path", default=[], help="Path to data folder")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = SortConfig()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0.get_all():
        default_key = ops0[k]
        args_key = dargs[k]
        if k in ["fast_disk", "save_folder", "save_path0"]:
            if args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        elif type(default_key) in [np.ndarray, list]:
            n = np.array(args_key)
            if np.any(n != np.array(default_key)):
                ops[k] = n.astype(type(default_key))
                print(set_param_msg.format(k, ops[k]))
        elif isinstance(default_key, bool):
            args_key = bool(int(args_key))  # bool("0") is true, must convert to int
            if default_key != args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        # checks default param to args param by converting args to same type
        elif not (default_key == type(default_key)(args_key)):
            ops[k] = type(default_key)(args_key)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


def main():
    # params path, not manual
    path = Path().home() / "cpsort"

    params = SortConfig(path)
    run(params, parallel=True, overwrite=False)


if __name__ == "__main__":
    main()
