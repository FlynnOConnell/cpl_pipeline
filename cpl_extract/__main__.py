import argparse
from pathlib import Path
from pprint import pprint

from icecream import ic

from cpl_extract import Dataset, load_dataset
from cpl_extract.defaults import defaults


def add_args(parser: argparse.ArgumentParser):
    """
    Adds arguments to parser.
    """
    parser.add_argument("-n", "--new", action="store_true", help="Create a new dataset.")
    parser.add_argument("-f", "--file", action="store_true", help="Load a dataset from an existing file.")
    parser.add_argument("-p", "--path", type=str, help="Base path containing raw or extracted data.")
    parser.add_argument("-P", "--parallel", action="store_true", help="Run in parallel.")
    parser.add_argument("-d", "--detect-spikes", action="store_true", help="Detect spikes.")
    parser.add_argument("-c", "--cluster-spikes", action="store_true", help="Cluster spikes.")
    parser.add_argument("-s", "--sort-spikes", action="store_true", help="Sort spikes.")
    parser.add_argument("-a", "--all", action="store_true", help="Run all steps.")

    for step in Dataset.PROCESSING_STEPS:
        parser.add_argument(f"--{step.replace(' ', '_')}", action="store_true", help=f"{step}")
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """ Parses arguments from parser. """

    args = parser.parse_args()
    dargs = vars(args)

    if args.path:
        datapath = Path(args.path).expanduser().resolve()
    else:
        datapath = Path(defaults()["datapath"]).expanduser().resolve()

    ic(f"Setting datapath to {datapath}")
    dargs["path"] = datapath

    if args.new:
        data = Dataset(root_dir=datapath, shell=True)
        dargs['data'] = data
    elif args.file:
        data = load_dataset(datapath, shell=True)
        dargs['data'] = data
    else:
        if list(Path(args.path).glob("*.p")):  # if it's a non-empty list
            data = load_dataset(args.path,)
        else:
            raise FileNotFoundError(f"No .p file found in {args.path}, please create a new dataset or load an existing one.")

    # Iterate over each processing step and execute if the corresponding flag is set
    for step in Dataset.PROCESSING_STEPS:
        arg = getattr(args, step.replace(' ', '_'))
        if arg:
            method_name = Dataset.PROCESSING_STEPS[step]
            if method_name:
                method = getattr(data, method_name)
                method()
            else:
                pprint(f"No method found for argument provided: {arg}"
                   f"Valid arguments: {Dataset.PROCESSING_STEPS.keys()}")

    return args, dargs


def main(shell=False):
    if shell:
        parser = argparse.ArgumentParser(description="Sorting pipeline parameters.")
        parser = add_args(parser)
        args, dargs = parse_args(parser)

        data = dargs['data']
        if not data.process_status['initialize_parameters']:
            data.initialize_parameters()

        if not data.process_status['extract_data']:
            data.extract_data()

        if not data.process_status['mark_dead_channels']:
            data.mark_dead_channels()

        if not data.process_status['spike_detection']:
            data.detect_spikes()

        if not data.process_status['spike_clustering']:
            data.cluster_spikes()

        if not data.process_status['cleanup_clustering']:
            data.cleanup_clustering()

        electrodes = data.electrode_mapping.loc[:, "electrode"].to_list()
        for electrode in electrodes:
            root, spikesort = data.sort_spikes(all_electrodes=electrodes)
            x = 5
        x = 5
    # else:
    #     filepath = Path().home() / "data" / 'serotonin' / '1'
    #     things = [x for x in filepath.glob('*')]
    #     file = [f for f in filepath.iterdir() if f.suffix == '.smr'][0]
    #     data = load_dataset(args.path, shell=True, )


if __name__ == "__main__":
    main(shell=True)
