# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

import math
from pathlib import Path

import tables

from cpl_extract import utils
from cpl_extract.sort.directory_manager import DirectoryManager
from cpl_extract.logger import logger
from cpl_extract.sort.sorter import ProcessChannel
from cpl_extract.sort.spk_config import SortConfig
from cpl_extract.sort.cluster import CplClust, SpikeDetector
from cpl_extract.spk_io.writer import read_dict_from_json


def run(params: SortConfig, parallel: bool = True, overwrite: bool = False):
    """
    Entry point for the clustersort package.
    Optionally include a `SpkConfig` object to override the default parameters.

    This function iterates over data files, manages directories, and executes sorting either sequentially
    or in parallel.

    Parameters
    ----------
    params : spk_config.SortConfig
        Configuration parameters for spike sorting. If `None`, default parameters are used.
    parallel : bool, optional
        Whether to run the sorting in parallel. Default is `True`.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is `False`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the run type specified in `params` is not either "Manual" or "Auto".

    Examples
    --------
    >>> sort(SortConfig(), parallel=True)
    """

    logger.info(f"{params.get_section('path')}")

    # TODO: Add a check for the number of files in the directory
    runpath = Path(params.get_section("path")["run"])
    data_files = list(runpath.glob("*.h5"))
    if len(data_files) == 0:
        logger.info(f"No files found in {runpath}")
        return
    num_cpu = int(params.run["cores-used"]) if parallel else 1
    # pbm.init_file_bar(len(runfiles))
    runfiles = data_files
    for curr_file in runfiles:
        if curr_file.suffix != ".h5":
            continue
        logger.info(f"Processing file: {curr_file}")
        # h5file = read_h5(curr_file)
        h5file = {"data": {}, "metadata_channel": {}, "channels": {}}
        all_data = h5file["channels"]

        # Extract only the unit data
        for key in ["VERSION", "CLASS", "TITLE"]:
            if key in all_data.keys():
                del all_data[key]

        unit_data = {}
        for key in all_data.keys():
            if utils.check_substring_content(key, "U"):
                unit_data[key] = all_data[key]
        num_chan = len(unit_data)

        # Create the necessary directories
        dir_manager = DirectoryManager(
            curr_file,
            num_chan,
            params,
        )
        dir_manager.flush_directories()
        dir_manager.create_base_directories()
        dir_manager.create_channel_directories()

        runs = math.ceil(num_chan / num_cpu)
        for n in range(runs):
            channels_per_run = num_chan // runs
            chan_start = n * channels_per_run
            chan_end = (n + 1) * channels_per_run if n < (runs - 1) else num_chan
            if chan_end > num_chan:
                chan_end = num_chan

            if parallel:
                raise NotImplementedError
            else:
                for i in range(chan_start, chan_end):
                    chan_name = [list(unit_data.keys())[i]][0]
                    chan_data = unit_data[chan_name]

                    sampling_rate = chan_data["metadata"]["fs"]
                    dir_manager.idx = i
                    # sort(
                    #     curr_file,
                    #     chan_data,
                    #     sampling_rate,
                    #     params,
                    #     dir_manager,
                    #     i,
                    #     overwrite=overwrite,
                    # )


def main():
    # my_data = Path("/media/thom/hub/data/serotonin/extracted/r11")
    logger.setLevel("INFO")
    my_data = Path().home() / "cpl_extract"

    main_params = SortConfig(my_data / "config.ini")
    main_params.save_to_ini()

    # dir_manager = DirectoryManager(
    #     my_data,
    #     4,
    #     main_params,
    # )
    #
    # dir_manager.flush_directories()

    h5 = my_data.glob("*.h5")
    curr_file = next(h5)
    save_path = curr_file.parent / curr_file.stem
    save_path.mkdir(parents=True, exist_ok=True)

    my_h5 = tables.open_file(str(curr_file), mode="r+")
    time_vector = my_h5.get_node("/raw_time/time_vector")
    chans = [node._v_name for node in my_h5.walk_nodes()]

    # only chans with "U" in the name are unit data
    chans_units = [chan for chan in chans if "U" in chan]

    for idx, chan in enumerate(chans_units):
        chan_data = my_h5.get_node(f"/raw_unit/{chan}")
        sampling_rate = chan_data._v_attrs["fs"]

        my_h5.close()
        param_path = Path('/home/thom/repos/cpl_extract/cpl_extract/sort/defaults/clustering_params.json')
        params = read_dict_from_json(str(param_path))
        sd = SpikeDetector(curr_file, chan, overwrite=True, params=params, fs=sampling_rate)
        sd.run()

if __name__ == "__main__":
    main()
