from pathlib import Path
import argparse

import logger
from sort.run_sort import run
from spk2extract.sort.spk_config import SortConfig
# import gui

def main():
    parser = argparse.ArgumentParser(description='Run spike sorting.')
    parser.add_argument('--path', type=str, help='Path to data files')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    logger.set_log_level("CRITICAL")
    main_params = SortConfig(Path(args.path))
    main_params.save_to_ini()
    run(main_params, parallel=args.parallel, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
