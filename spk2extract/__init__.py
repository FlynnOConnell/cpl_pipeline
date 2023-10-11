"""
===============
``spk2extract``
===============

*spk2extract. A spike 2 data extraction utility for extracellular recordings.*

Documentation Guide
-------------------

I recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.

"""
try:
    from platformdirs import user_cache_dir, user_config_dir, user_log_dir
except ImportError:
    user_cache_dir = None
    user_config_dir = None
    user_log_dir = None
    pass
from pathlib import Path
from . import spk_io, gui, helpers, defaults

__name__ = "spk2extract"
__author__ = "Flynn OConnell"
__all__ = [
    "spk_io",
    "gui",
    "helpers",
    "defaults",
]

# Version
version = "0.0.1"

# Platform-dependent directories
def _init_directories():
    spk2dir = Path().home() / "spk2extract"
    if not spk2dir.exists():
        spk2dir.mkdir(exist_ok=True)

    # make sure platformdirs import was successful
    if user_cache_dir and user_config_dir and user_log_dir:
        return {
            "cache_dir": user_cache_dir(__name__, __author__),
            "config_dir": user_config_dir(__name__, __author__),
            "log_dir": user_log_dir(__name__, __author__),
            "spk2dir": spk2dir,
        }
    else:
        return {}

def get_logger():
    from .logs import logger
    return logger

directories = _init_directories()
__all__.extend(list(directories.keys()))
globals().update(directories)
