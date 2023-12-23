# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import logging
from pathlib import Path
from cpl_extract import *

# from cpl_extract import load_dataset, detect_number_of_cores
#
# from cpl_extract.base.dataset import Dataset
# from cpl_extract.base.objects import *
#
# from cpl_extract import load_project, load_experiment, experiment

if "CACHE_PATH" in os.environ:
    cache_path = Path(os.environ["CPL_CACHE_PATH"])
else:
    cache_path = Path().home() / ".cache"

N_CORES = detect_number_of_cores()

def _sort():
    logging.info("Initializing CPL Extract")
    root_dir = Path().home() / 'data' / 'serotonin'

    datasets = []
    for rec_dir in iter_dirs(root_dir):
        filename = [f for f in rec_dir.iterdir() if f.suffix == '.smr'][0]
        data = Dataset(rec_dir, filename.stem)
        data.initialize_parameters(shell=True, accept_params=True)
        data.extract_data()
        datasets.append(data)

def iter_dirs(data_dir):
    for d in data_dir.iterdir():
        if d.is_dir():
            yield d

def main():
    filepath = Path().home() / "data" / 'serotonin' / '1'
    things = [x for x in filepath.glob('*')]
    file = [f for f in filepath.iterdir() if f.suffix == '.smr'][0]
    data = load_dataset(filepath, shell=True,)
    # data = Dataset(filepath, file.stem)
    # data.initParams(shell=True, accept_params=True)
    # data.extract_data()
    # data.detect_spikes()
    # data.cleanup_clustering()
    data.sort_spikes()

def setup_proj():
    root = Path().home() / "data" / "serotonin"
    proj = experiment(root)
    return proj

def debug():
    x = 2

if __name__ == "__main__":
    # main()
    debug()