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
from platformdirs import *

from spk2extract.defaults import defaults  # noqa (API import)
from spk2extract.extraction import SpikeData, UnitData  # noqa (API import)
from spk2extract.gui import *  # noqa (API import)
from spk2extract.spk_io import * # noqa (API import)
from spk2extract.util import *  # noqa (API import)
from spk2extract.version import version as __version__  # noqa (API import)

version = __version__
__name__ = "spk2extract"
__author__ = "Flynn OConnell"

# Platform-dependent directories
spk2dir = Path().home() / "spk2extract"
if not spk2dir.exists():
    spk2dir.mkdir(exist_ok=True)
cache_dir = user_cache_dir(__name__, __author__)  # Cache, temp files
config_dir = user_config_dir(__name__, __author__)  # Config, parameters and options
log_dir = user_log_dir(__name__, __author__)  # Logs, .log files primarily

# Documentation inclusions
__all__ = [
    "SpikeData",
    "UnitData",
    "spk_io",
    "extract_waveforms",
    "dejitter",
    "filter_signal",
    "defaults",
    "version",
    "gui",
    "cache_dir",
    "config_dir",
    "log_dir",
    "spk2dir",
]
x = 5