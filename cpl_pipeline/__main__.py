import argparse
from pathlib import Path

from icecream import ic

from cpl_pipeline import Dataset, load_dataset, load_pickled_object
from cpl_pipeline.defaults import defaults


def add_args(parser: argparse.ArgumentParser):
    """
    Adds arguments to parser.
    """

    parser.add_argument("-n", "--new", nargs='?', const=True, type=str,
                        help="Create a new dataset, optionally specify a path.")
    parser.add_argument("-f", "--file", nargs='?', const=True, type=str,
                        help="Load a dataset from an existing file, optionally specify a path.")

    parser.add_argument("-y", "--yes", action="store_true", help="Auto-accept all arguments.")
    parser.add_argument("-P", "--parallel", action="store_true", help="Run in parallel.")
    parser.add_argument("-a", "--all", action="store_true", help="Run all steps.")

    for step in Dataset.PROCESSING_STEPS:
        parser.add_argument(f"--{step.replace(' ', '_')}", action="store_true", help=f"{step}")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """ Parses arguments from parser. """

    args = parser.parse_args()

    if args.new:
        ic(f"Creating new dataset at {args.new}")
        datapath = Path(args.new).expanduser().resolve()
        data = Dataset(datapath)
        if args.yes:
            data.initialize_parameters(accept_params=True)
            data.extract_data()
        else:
            data.initialize_parameters()
    elif args.file:
        ic(f"Loading dataset from {args.file}")
        datapath = Path(args.file).expanduser().resolve()
        data = load_dataset(datapath)
    else:
        datapath = Path(defaults()["datapath"]).expanduser().resolve()
        if datapath.is_dir():
            data = Dataset(datapath)
        else:
            data = load_dataset(datapath)

    # iterate over each processing step and execute if the corresponding flag is set
    if args.all:
        data.pre_process_for_clustering("initialize_parameters", accept_params=args.yes,)
    else:
        for i, step in enumerate(Dataset.PROCESSING_STEPS):
            arg = getattr(args, step.replace(' ', '_'))
            if arg:
                method_name = Dataset.PROCESSING_STEPS[i]
                if method_name:
                    ic(f"Calling {method_name} from CLI args.")
                    method = getattr(data, method_name)
                    method()
                else:
                    ic(f"No method found for argument provided: {arg}"
                           f"Valid arguments: {Dataset.PROCESSING_STEPS.keys()}")
    return args, data


def main(shell=False):
    if shell:
        parser = argparse.ArgumentParser(description="Sorting pipeline parameters.")
        args, data = parse_args(add_args(parser))
    else:
        from cpl_pipeline import load_dataset

        data = Dataset(Path().expanduser().resolve())
        data.initialize_parameters()
        data.extract_data()


if __name__ == "__main__":
    main(shell=True)
