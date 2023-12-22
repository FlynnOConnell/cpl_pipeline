import argparse
from pathlib import Path

from icecream import ic, install

from cpl_extract.defaults import defaults
from cpl_extract import run_sort, Dataset, load_dataset, detect_number_of_cores


def add_args(parser: argparse.ArgumentParser):
    """
    Adds arguments to parser.
    """
    parser.add_argument("-n", "--new", action="store_true", help="Create a new dataset.")
    parser.add_argument("-f", "--file", action="store_true", help="Load a dataset from an existing file.")
    parser.add_argument("-v", "--version", action="store_true", help="Print version.")
    parser.add_argument("-p", "--path", type=str, help="Base path containing raw or extracted data.")
    parser.add_argument("-P", "--parallel", action="store_true", help="Run in parallel.")
    parser.add_argument("-d", "--detect-spikes", action="store_true", help="Detect spikes.")
    parser.add_argument("-c", "--cluster-spikes", action="store_true", help="Cluster spikes.")
    parser.add_argument("-s", "--sort-spikes", action="store_true", help="Sort spikes.")
    parser.add_argument("-a", "--all", action="store_true", help="Run all steps.")

    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    set_param_msg = "->> Setting {0} to {1}"
    if args.version:
        ic(f"Version: 0.0.2")  # TODO: Get version from __init__.py

    if args.new:
        data = Dataset(root_dir=args.path, shell=True)
        dargs['data'] = data
    elif args.file:
        data = load_dataset(args.path, shell=True)
        dargs['data'] = data
    else:
        raise ValueError("No dataset specified.")
    if args.path:
        datapath = Path(args.path).expanduser().resolve()
        ic(f"Setting datapath to {datapath}")
        dargs["path"] = datapath
    if args.parallel:
        ic("Setting parallel to True")
        dargs["parallel"] = True
    return args, dargs


def main(shell=False):
    if shell:
        parser = argparse.ArgumentParser(description="Sorting pipeline parameters.")
        parser = add_args(parser)
        args, dargs = parse_args(parser)

        data = dargs['data']
        if not data.process_status['initialize parameters']:
            data.initParams()
        if not data.process_status['mark_dead_channels']:
            data.mark_dead_channels()
        if not data.process_status['spike_detection']:
            data.detect_spikes()
        if not data.process_status['spike_clustering']:
            data.cluster_spikes()
        if not data.process_status['cleanup_clustering']:
            data.cleanup_clustering()

        if args.detect_spikes:
            if args.parallel:
                ic(detect_number_of_cores())
                data.detect_spikes(parallel=True, n_cores=detect_number_of_cores() / 2)
            else:
                data.detect_spikes()
        if args.cluster_spikes:
            data.cluster_spikes()
        if args.sort_spikes:
            data.sort_spikes()

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
