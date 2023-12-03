import argparse
import numpy as np
from spk2extract import version
from spk2extract.defaults import defaults


def add_args(parser: argparse.ArgumentParser):
    """
    Adds config options arguments to parser.
    """
    parser.add_argument("--version", action="version", version=version)
    parser.add_argument("--sort", default=None, help="Run autosort pipeline")
    parser.add_argument('--extract', default=None, help='Run extract pipeline')
    parser.add_argument('--ops', default=None, help='path to options.npy file')
    parser.add_argument('--test', default=None, help='Run tests on pipeline')
    parser.add_argument('--gui', default=None, help='Open GUI')
    parser.add_argument('--verbose', default=None, help='Display help information')
    parser.add_argument(
        "--config", default=defaults(), help="Path to configuration .ini file, including filename."
    )
    parser.add_argument("--path", default=[], help="Path to data folder")
    return parser

def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = defaults()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    config = np.load(args.config, allow_pickle=True).item() if args.config else {}
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

def test(parser: argparse.ArgumentParser):
    parser = argparse.ArgumentParser()
    parser.add_argument("ops", type=str, default=None)
    parser.add_argument("test", type=str, default=None)
    parser.add_argument("-v", "--verbose", action='count', type=str, help= "Provides a detailed description of the argument")

    args: argparse.Namespace = parser.parse_args()
    print(args)

def main():
    from spk2extract import gui
    gui.run()

if __name__ == "__main__":
    main()
