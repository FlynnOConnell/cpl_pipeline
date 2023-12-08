"""
Configuration module for the clustersort package.
"""
from __future__ import annotations
import requests
import configparser
from pathlib import Path
from typing import Any


def download_default_config(path_to_save: Path | str = None):
    """
    Downloads the default configuration file from the github repository.
    """
    default_config_url = (
        "https://github.com/FlynnOConnell/clustersort/raw/master/default_config.ini"
    )
    try:
        response = requests.get(default_config_url)
        response.raise_for_status()
        with open(path_to_save, "wb") as file:
            file.write(response.content)
        print(f"Default configuration file downloaded to {path_to_save}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download the default configuration file: {e}")


class SortConfig:
    """
    Initialize a new SpkConfig object to manage configurations for the cpsort pipeline.

    If no config path is specified, the following occurs:
    1) Checks your base_path (normally users/home/sort) for any .ini files.
        If none are found, a default configuration file is created.

    2) If a default configuration file is found, it is copied to your base_path.
        If no default configuration file is found, one is created and copied to your base_path.

    3) The configuration file is read and stored in a ConfigParser object.

    Parameters
    ----------
    cfg_path : str or Path, optional
        The path to the configuration file. Defaults to a pre-defined location.

    Notes
    -----
    The configuration file is an INI-style file with the following sections:
    -base: str
        Contains the base path for the cpsort pipeline. This is where results are saved, data and figures.
    - path : dict
        Contains paths for the data and figures produced and used by the cpsort pipeline.
    - run: dict
        Contains configurations related to runtime settings like 'resort-limit', 'cores-used'
    - cluster : dict
        Contains clustering parameters like 'max-clusters', 'max-iterations'.
    - breach : dict
        Contains breach analysis parameters like 'disconnect-voltage', 'max-breach-rate'.
    - filter : dict
        Contains filter parameters like 'low-cutoff', 'high-cutoff'.
    - spike : dict
        Contains spike-related settings like 'pre-time', 'post-time'.

    .. note::
       Due to the nature of INI files, all values are stored as strings. It is up to the user
       to convert the values to the appropriate type.

    Examples
    --------
    >>> cfg = SortConfig()
    >>> run = cfg.run
    >>> print(type(run), run)
    <class 'dict'> {'resort-limit': '3', 'cores-used': '8', ...}

    >>> cfg.set('run', 'resort-limit', 5)
    >>> print(cfg.run['resort-limit'])
    '5'

    See Also: `configparser from python std library <https://docs.python.org/3/library/configparser.html>`_

    """

    def __init__(self, cfg_path: Path | str):
        """
        Initialize a new SpkConfig object to manage configurations for the AutoSort pipeline.

        Parameters
        ----------
        cfg_path : str or Path, optional
            The path to the configuration file. Defaults to the repository config.
        """
        self.cfg_path = Path(cfg_path)
        if self.cfg_path.is_dir():
            # if the provided path is a directory, append the filename
            config_file = self.cfg_path.glob("*.ini")
            num_config_files = len(list(config_file))
            if num_config_files > 1:
                self.cfg_path = self.cfg_path / "default_config.ini"
                download_default_config(self.cfg_path)
            elif num_config_files > 1:
                raise Exception(
                    f"Multiple configuration files found in {self.cfg_path}"
                )
        elif not self.cfg_path.is_file():
            self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
            download_default_config(self.cfg_path)

        self.config = self.read_config()
        self.all_params = self.get_all()
        self._validate_config()

    def __getitem__(self, item):
        return self.all_params[item]

    def __setitem__(self, section, key, value):
        self.set(section, key, value)

    def get_section(self, section: str):
        """
        Returns a dictionary containing key-value pairs for the given section.

        Parameters
        ----------
        section : str
            The name of the section in the config file.

        Returns
        -------
        dict
            Dictionary containing the section's key-value pairs.
        """
        return dict(self.config[section])

    def set(self, section: str, key: str, value: Any):
        """
        Set the value of a configuraton parameter.

        Parameters
        ----------
        section : str
            The section of the paremeter being set.
        key : str
            The key of the section to be set.
        value : str
            The value being set by the user.

        Returns
        -------
        None

        .. note::

            All parts of the configuration, section, key and value, are strings. This is how configparser works
            under the hood. It is up to the user to convert datatypes when needed.
        """

        if section not in self.config:
            self.config.add_section(section)
        self.config.set(section, key, str(value))

    def get_all(self):
        """
        Get a dictionary of ``key: value`` pairs for every section, conglomerated.

        Returns
        -------
        dict
            A dictionary containing all key-value pairs from all sections.

        """
        params = {}
        for section in self.config.sections():
            for key, value in self.config.items(section):
                params[key] = value
        return params

    @property
    def run(self):
        """
        Get a dictionary containing all ``key: value`` pairs for the ``run`` section.

        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'run' section.
        """
        return self.get_section("run")

    @property
    def path(self):
        """
        Get a dictionary containing all ``key: value`` pairs for the ``path`` section.

        The path section stores paths for the plots and data produced and used by the clustersorting pipeline.

        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'path' section.
        """
        return self.get_section("path")

    @property
    def cluster(self):
        """
        Get a dictionary containing all ``key: value`` pairs for the ``cluster`` section.

        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'cluster' section.
        """
        return self.get_section("cluster")

    @property
    def breach(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'breach' section.
        """
        return self.get_section("breach")

    @property
    def filter(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'breach' section.
        """
        return self.get_section("filter")

    @property
    def spike(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'spike' section.
        """
        return self.get_section("spike")

    @property
    def detection(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'detection' section.
        """
        return self.get_section("detection")

    @property
    def pca(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'pca' section.
        """
        return self.get_section("pca")

    @property
    def postprocess(self):
        """
        Returns
        -------
        dict
            A dictionary containing key-value pairs for the 'postprocess' section.
        """
        return self.get_section("postprocess")

    def read_config(self):
        """
        Returns
        -------
        ConfigParser
            A ConfigParser object loaded with the INI file.
        """
        config = configparser.ConfigParser()
        config.read(self.cfg_path)
        return config

    def reload_from_ini(self):
        self.config = self.read_config()
        self.all_params = self.get_all()

    def save_to_ini(self):
        with open(self.cfg_path, "w") as configfile:
            self.config.write(configfile)

    def _validate_config(self):
        """
        Validates the loaded configurations to ensure they meet specified criteria.

        Raises
        ------
        AssertionError
            If any of the loaded configurations are not valid.
        """
        assert (
            self.cfg_path.is_file()
        ), f"Configuration file {self.cfg_path} does not exist"
        assert self.run["run-type"] in ["Auto", "Manual"], (
            f"Run type {self.run['run-type']} is not valid. Options "
            f"are 'Auto' or 'Manual'"
        )
