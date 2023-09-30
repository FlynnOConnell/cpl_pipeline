import argparse
import numpy as np
from spk2extract import defaults, version


def add_args(parser: argparse.ArgumentParser):
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = defaults()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0:
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
    # # args, ops = parse_args(
    #     add_args(argparse.ArgumentParser(description="spk2extract parameters")))
    # if args.version:
    #     print("spk2extract v{}".format(version))
    # elif args.single_plane and args.ops:
    #     pass
    # elif len(args.db) > 0:
    #     db = np.load(args.db, allow_pickle=True).item()
    #     # from spk2extract import run_s2e
    #     # run_s2p(ops, db)
    #     print("not implemented yet")
    #     pass
    # else:
    from spk2extract import gui
    gui.run()


if __name__ == "__main__":
    main()