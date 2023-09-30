"""
Class for managing directories produced and utilized when running the clustersort pipeline.

"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path

class DirectoryManager:
    """
    Manages directories.

    .. note::

        Several temporary and permanent directories must be made so save plotting results, and to
        store data used to create those plots.

    """

    @staticmethod
    def find_files_by_extension(ext: str='.*', max_depth=3) -> list[Path]:
        path = Path.home()
        pattern = '/'.join(['*'] * max_depth)
        glob_pattern = f"{pattern}/*.{ext}"
        return list(path.glob(glob_pattern))


    def __init__(self, filepath: str | Path):
        """
        Parameters
        ----------
        filepath : str or Path
            Full filepath of the data file being processed.

        Returns
        -------
        None

        .. note::
            Must delete some directories.

        """
        self.filename = Path(filepath).stem
        self.base_path = Path(filepath).parent / self.filename
        self.base_suffix = self.base_path.suffix
        self.raw = self.base_path / "raw"
        self.h5 = self.base_path / "h5"
        self.merged = self.base_path / "merged"
        self.directories = [
            self.raw,
            self.h5,
            self.merged,
        ]
        self.idx = 0
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())

    def create_base_directories(self):
        """
        Creates the base directories for raw data, processed data, plots, reports, and temporary files.
        """
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
                    self.logger.debug(f"Found base_dir: {f}")
                    if f.is_file():
                        self.logger.debug(f"Deleting file: {f}")
                        f.unlink()
                    elif f.is_dir():
                        self.logger.debug(f"Deleting directory: {f}")
                        shutil.rmtree(f)
        except Exception as e:
            self.logger.error(f"Error flushing directories: {e}", exc_info=True)
