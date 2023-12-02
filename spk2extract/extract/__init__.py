"""
Spike extract module for extracellular recordings.

Current supported formats:
    - Spike2 (.smr)
    - Plexon (.plx)

This uses sonpy for spike2 rather than Neo. Neo is a great package, but it makes extracting events impossible from
spike2 files.

"""
from .spike2 import Spike2Data
from spk2extract.viz.plots import *

__all__ = ["Spike2Data", "plot_coh",]
