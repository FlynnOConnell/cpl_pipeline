# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from cpl_extract import logger
from cpl_extract.base.dataset import Dataset
from cpl_extract.base.objects import load_dataset

def main():
    logger.use_log_level("INFO")
    print("Starting main_dataset")

    root_dir = Path().home() / "cpl_extract"
    data_dir = Path("/media/thom/hub/data/serotonin/raw/")

    animal = list(data_dir.glob("*.smr"))[0]

    data = Dataset(root_dir=root_dir, data_dir=data_dir, data_name=animal.stem,)
    data.initParams(accept_params=True)
    data.extract_data()
    # data.mark_dead_channels()
    # data = load_dataset(root_dir, data_dir,)
    # print(data)

    data.detect_spikes()
    # data.blech_clust_run(multi_process=True, n_cores=16, umap=False, accept_params=True)
    # data.sort_spikes(3)

if __name__ == "__main__":
    main()
