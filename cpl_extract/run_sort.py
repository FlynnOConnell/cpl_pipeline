# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path

from cpl_extract import load_dataset
from cpl_extract.base.dataset import Dataset

log_file = Path().home() / "cpl_extract"/ "logs" / "base.log"
log_file.parent.mkdir(exist_ok=True)
log_file.touch(exist_ok=True)

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def _init_data():
    logging.info("Initializing CPL Extract")

    root_dir = Path().home() / "cpl_extract"
    data_dir = Path("/media/thom/hub/data/serotonin/raw/")

    data = Dataset(root_dir, data_dir)
    data.initParams(shell=True, accept_params=True)
    return data

def _extract_data(data):
    data.extract_data()
    return data

def _detect_spikes(data):
    data.detect_spikes()
    return data

def _blech_clust_run(data):
    data.blech_clust_run()
    return data

def _sort_spikes(data):
    data.sort_spikes(3)
    return data

def main():
    data = load_dataset(Path().home() / "cpl_extract")
    _, ss_gui = data.sort_spikes()
    ss_gui.mainloop()

if __name__ == "__main__":
    main()


