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
from .extraction import SpikeData, UnitData # noqa (API import)
from .util.spike_io import read_h5, write_h5 # noqa (API import)
from .util.cluster import filter_signal, extract_waveforms, dejitter # noqa (API import)

__version__ = "0.1.0"