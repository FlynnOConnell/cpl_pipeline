# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path

from cpl_extract import load_dataset, detect_number_of_cores

from cpl_extract.base.dataset import Dataset
from cpl_extract.base.objects import *

from cpl_extract import load_project, load_experiment, experiment


log_file = Path().home() / "cpl_extract"/ "logs" / "base.log"
log_file.parent.mkdir(exist_ok=True)
log_file.touch(exist_ok=True)

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

N_CORES = detect_number_of_cores()

def _sort():
    logging.info("Initializing CPL Extract")
    home = Path().home()
    root_dir = Path().home() / 'data' / 'serotonin'

    datasets = []
    for rec_dir in iter_dirs(root_dir):
        filename = [f for f in rec_dir.iterdir() if f.suffix == '.smr'][0]
        data = Dataset(rec_dir, filename.stem)
        data.initParams(shell=True, accept_params=True)
        data.extract_data()
        datasets.append(data)

def iter_dirs(data_dir):
    for d in data_dir.iterdir():
        if d.is_dir():
            yield d

def main():
    filepath = Path().home() / "data" / "serotonin" / "5"
    data = load_dataset(filepath)
    _, ss_gui = data.sort_spikes()
    ss_gui.mainloop()

def setup_proj():
    root = Path().home() / "data" / "serotonin"
    project = experiment(root)
    return project


if __name__ == "__main__":
    main()

    x = 5