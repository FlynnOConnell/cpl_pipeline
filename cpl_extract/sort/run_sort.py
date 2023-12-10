# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations
from pathlib import Path
from cpl_extract import logger
# from cpl_extract.sort.cluster import CplClust, SpikeDetector
# from cpl_extract.spk_io.writer import read_dict_from_json
from cpl_extract.base.dataset import Dataset
# from cpl_extract.spk_io.printer import print_globals_and_locals, print_differences

if __name__ == "__main__":
    logger.set_log_level("info")
    logger.cpl_logger.info("Starting main_dataset")

    root_dir = Path().home() / "cpl_extract"
    data_dir = Path("/media/thom/hub/data/serotonin/raw/")

    animal = list(data_dir.glob("*.smr"))[0]

    data = Dataset(root_dir=root_dir, data_dir=data_dir, data_name=animal.stem,)
    data.initParams(accept_params=True)
    data.extract_data(animal, root_dir)
    data.mark_dead_channels()
    data.detect_spikes()
