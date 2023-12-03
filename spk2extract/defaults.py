from .version import version
from pathlib import Path

def defaults():
    """default options to run spike extract"""
    return {
        "spk2extract_version": version,  # current version of package
        "data_path": Path().home() / "data",
        "cache_path": Path().home() / "data" / ".cache",
        "log_path": Path().home() / "data" / "logs",  # path to save logs
        "log_level": "INFO",  # level of logging
        "log_file": "cp_sort.log",  # name of log file
        "log_file_level": "INFO",  # level of logging to file
    }
