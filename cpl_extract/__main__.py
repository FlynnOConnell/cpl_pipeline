import argparse
from pathlib import Path

from icecream import ic, install

from cpl_extract.defaults import defaults
from cpl_extract import run_sort, Dataset, load_dataset, detect_number_of_cores

install()


def add_args(parser: argparse.ArgumentParser):
    """
    Adds arguments to parser.
    """
    parser.add_argument("-d", action="store_true", help="Supply a path to the dataset.")
    parser.add_argument("-p", "--path", type=str, help="Path to dataset.")
    parser.add_argument("-v", "--version", action="store_true", help="Print version.")
    parser.add_argument("--shell", action="store_true", help="Run in shell mode.")
    return parser

def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    set_param_msg = "->> Setting {0} to {1}"
    if args.version:
        print(f"Version: 0.0.2")  # TODO: Get version from __init__.py
        ic('Version: 0.0.2')
    if args.shell:
        ic('Running in shell mode.')
    return args, dargs

def main():
    parser = argparse.ArgumentParser(description="Run spike sorting.")
    parser.add_argument("-p", "--path", type=str, help="Path to data files")
    parser.add_argument("-c", "--config", type=str, help="Path to config ini")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("-l", "--loglevel", action="store_true", help="Overwrite existing files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--sort", action="store_true", help="Run sorting")
    args = parser.parse_args()

    ops = defaults()
    if args.path is None:
        args.path = ops["data_path"]
    if args.config is None:
        args.config = ops["config_path"]
    if args.overwrite is None:
        args.overwrite = ops["overwrite"]
    if args.loglevel is None:
        args.loglevel = ops["loglevel"]
    if args.parallel is None:
        args.parallel = ops["parallel"]
    if args.verbose is None:
        args.verbose = ops["verbose"]


def main_test():
    run_sort.main()
    args, dargs = parse_args(
        add_args(argparse.ArgumentParser(description="Sorting pipeline parameters.")))
    if args.path or args.p:
        datapath = Path(args.path)
    else:
        raise ValueError("No path supplied.")
    if args.dataset:
        filepath = Path().home() / "data" / '1'
        file = [f for f in filepath.iterdir() if f.suffix == '.smr'][0]
        data = Dataset(filepath, file.stem)
        data.initParams(shell=True, accept_params=True)
        data.extract_data()


if __name__ == "__main__":
    main()
