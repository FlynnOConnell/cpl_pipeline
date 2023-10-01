"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu
from pkg_resources import iter_entry_points

from . import io
from . import rungui
from spk2extract.spk_io.utils import get_spk2extract_path
from spk2extract.extraction import SpikeData
from spk2extract import SpikeData


def mainmenu(parent):
    main_menu = parent.menuBar()

    # load folder of processed data
    loadFolder = QAction("&Load raw .smr file", parent)
    loadFolder.setShortcut("Ctrl+L")
    loadFolder.triggered.connect(lambda: io.load_smr(parent))
    parent.addAction(loadFolder)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(loadFolder)

def load_raw_spike2(filepath):
    s2 = SpikeData(filepath,)

def run_s2e(parent):
    RW = rungui.RunWindow(parent)
    RW.show()
