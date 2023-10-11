from .version import version
from pathlib import Path

def defaults():
    """default options to run spike extraction"""
    return {
        "spk2extract_version": version,  # current version of package
        "data_path": Path().home() / "data",  # path to data
        "save_path": Path().home() / "data" / "extracted",  # path to save data
        "log_path": Path().home() / "data" / "logs",  # path to save logs
        "log_level": "INFO",  # level of logging
        "log_file": "spike_extraction.log",  # name of log file
        "log_file_level": "DEBUG",  # level of logging to file
    }
