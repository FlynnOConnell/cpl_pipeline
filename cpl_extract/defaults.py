from icecream import ic

from .version import version
from pathlib import Path


def defaults():
    """
    fallback default path options to start the pipeline
    """
    base_path = Path().home() / 'data'
    return {
        "cpl_pipeline_version": version,  # current version of package
        "base_path": base_path,  # base path for all files
        "log_path": base_path / ".logs",  # path to save log files
        "data_quality": "noisy",  # data quality to use for sorting
        "log_level": "INFO",  # level of logging
        "overwrite": False,  # overwrite existing files
        "parallel": False,  # run in parallel
        "verbose": False,  # verbose output
    }


if __name__ == "__main__":
    d = defaults()
    ic(d)
