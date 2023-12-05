from .version import version
from pathlib import Path

def defaults():
    """default options to run spike extract"""
    home = Path().home()
    cpe_path = home / "spk2extract"
    cpe_path.mkdir(exist_ok=True)

    return {
        "spk2extract_version": version,  # current version of package
        "data_path": cpe_path / "data",  # path to save data
        "config_path": cpe_path / "config.ini",  # path to save config file
        "sort_path": cpe_path / "sorted",  # path to save sort files
        "log_path": cpe_path / "logs",  # path to save log files
        "log_level": "INFO",  # level of logging
        "log_file": "cp_sort.log",  # name of log file
        "log_file_level": "INFO",  # level of logging to file
    }
