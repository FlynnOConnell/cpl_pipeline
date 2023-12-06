"""
===============
``cpl_extract``
===============

*cpl_extract. A utility for extraction and sorting of extracellular recordings.*

"""
try:
    from platformdirs import user_cache_dir, user_config_dir, user_log_dir
except ImportError:
    user_cache_dir = None
    user_config_dir = None
    user_log_dir = None
    pass
from pathlib import Path
from . import spk_io, gui, utils, defaults, sort, base

__name__ = "cpl_extract"
__author__ = "Flynn OConnell"
__all__ = ["spk_io", "gui", "utils", "defaults", "sort", "base"]

# Version
version = "0.0.1"

# Platform-dependent directories
def _init_directories():
    cpe_dir = Path().home() / "cpl_extract"
    if not cpe_dir.exists():
        cpe_dir.mkdir(exist_ok=True)

    # use pre-existing dirs if possible
    if user_cache_dir and user_config_dir and user_log_dir:
        return {
            "cache_dir": user_cache_dir(__name__),
            "config_dir": user_config_dir(__name__),
            "log_dir": user_log_dir(__name__),
            "cpe_dir": cpe_dir,
        }
    else:
        return {
            "cpe_dir": cpe_dir,
            "cache_dir": cpe_dir / "cache",
            "config_dir": cpe_dir / "config",
            "log_dir": cpe_dir / "logs",
        }


def run_pipeline():
    pass


directories = _init_directories()
__all__.extend(list(directories.keys()))
globals().update(directories)
