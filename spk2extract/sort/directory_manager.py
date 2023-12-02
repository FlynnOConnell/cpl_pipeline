"""
=================
Directory Manager
=================

Class for managing directories produced and utilized when running the clustersort pipeline.

"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from logger import logger
from spk_config import SortConfig


class DirectoryManager:
    """
    Manages directory creation, destruction and status per-file for storing processed data, plots, reports, and intermediate files based
    off of the provided "base" filepath.

    Parameters
    ----------
    filepath : str or Path
        The base path for saving plots and intermediate data.
    base_path : str or Path
        The base path for saving plots and intermediate data. This will be the parent directory for
        all data files, plots, and reports. Default is Path().home() / "clustersort", likely
        set from from SortConfig() class.

    Notes
    -----
    Temporary files are cached according to your operating system.
        - On Windows, this is typically in the AppData/Local/Temp directory.
        - On Linux, this is typically in the /tmp directory.
        - On macOS, this is typically in the /var/folders directory.

    """

    def __init__(self, filepath: str | Path, num_chan, params: SortConfig):
        """
        Parameters
        ----------
        filepath : str or Path
            Full filepath of the data file being processed.
        """
        self.filename = Path(filepath).stem
        self.filetype = Path(filepath).suffix  # .h5, .nwb, .npy
        self.params = params
        self.base_path = params.cfg_path.parent / self.filename
        self.directories = [
            self.plots,
            self.reports,
            self.data,
        ]
        self.num_channels = num_chan
        self.idx = 0  # not used, could hold index for sorting each channel

        self.status_path = self.base_path / "status.npy"
        self.overwrite = params.get_section("run")["overwrite"]
        self.min_clusters = int(params.get_section("cluster")["min-clusters"])
        self.max_clusters = int(params.get_section("cluster")["max-clusters"])
        # Channel x Cluster boolean array
        self.status_data = np.zeros(
            (self.num_channels, (self.max_clusters - 1)),
            dtype=bool,
        )
        self.load_status()

    def check_processed(
        self,
):
        # If a channel is missing, just redo all of them (for now)
        return np.any(self.status_data == False)

    def load_status(self):
        if self.status_path.is_file():
            logger.info(f"Loading status file: {self.status_path}")
            self.status_data = np.load(self.status_path)
        else:
            logger.warning(f"Status file not found: {self.status_path}")

    def save_status(self, channel, cluster):
        # Account for 0 indexing here, rather than in main script
        channel_idx = channel - 1
        cluster_idx = cluster - 1
        self.status_data[channel_idx, cluster_idx] = True
        np.save(self.status_path, self.status_data)

    def should_process(
        self,
        channel,
        cluster,
    ):
        if self.overwrite:
            return True
        return not self.status_data[channel, cluster]

    @property
    def plots(self):
        """
        Path : Directory for storing plots.
        """
        return self.base_path / "Plots"

    @property
    def reports(self):
        """
        Path : Directory for storing reports.
        """
        return self.base_path / "Reports"

    @property
    def data(self):
        """
        Path : Directory for storing intermediate files.
        """
        return self.base_path / "Data"

    @property
    def channel(self):
        """
        int : Channel index incremented by 1.
        """
        return self.idx + 1

    def create_base_directories(self):
        """
        Creates the base directories for cached data, plots, reports, and temporary files.
        """
        logger.info(f"Creating base directories: {self.directories}")
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def flush_directories(self):
        """
        Deletes all files and subdirectories in each base directory.

        Raises
        ------
        Exception
            If there is an error in deleting files or directories.
        """
        try:
            for base_dir in self.directories:
                for f in base_dir.glob("*"):
                    logger.debug(f"Found base_dir: {f}")
                    if f.is_file():
                        logger.debug(f"Deleting file: {f}")
                        f.unlink()
                    elif f.is_dir():
                        logger.debug(f"Deleting directory: {f}")
                        shutil.rmtree(f)
        except Exception as e:
            logger.error(f"Error flushing directories: {e}", exc_info=True)

    def create_channel_directories(self):
        """
        Creates a set of subdirectories for a specific channel under each base directory.
        """
        for base_dir in self.directories:
            for channel_number in range(1, self.num_channels + 1):
                channel_dir = base_dir / f"channel_{channel_number}"
                logger.debug(f"Creating channel directory: {channel_dir}")
                channel_dir.mkdir(parents=True, exist_ok=True)
