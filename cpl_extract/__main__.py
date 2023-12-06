from pathlib import Path
import argparse

import logger
from sort.run_sort import run
from cpl_extract.sort.spk_config import SortConfig
from cpl_extract.defaults import defaults


def main():
    parser = argparse.ArgumentParser(description="Run spike sorting.")
    parser.add_argument("-p", "--path", type=str, help="Path to data files")
    parser.add_argument("-c", "--config", type=str, help="Path to config ini")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "-l", "--loglevel", action="store_true", help="Overwrite existing files"
    )
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

    logger.set_log_level(verbose=args.loglevel)
    config = SortConfig(Path(args.config))  # the config contains the path to the data
    if args.sort:
        run(config, parallel=args.parallel, overwrite=args.overwrite)
    else:
        pass
        # gui.run(config)


if __name__ == "__main__":
    main()
