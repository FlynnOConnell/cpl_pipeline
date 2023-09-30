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
from .extraction import SpikeData, UnitData     # noqa (API import)
from .util import *                             # noqa (API import)
from .gui import *                              # noqa (API import)

# Documentation inclusions
__all__ = ["SpikeData", "UnitData", "extract_waveforms", "dejitter", "filter_signal", "write_h5", "read_h5",]

__version__ = "0.1.0"