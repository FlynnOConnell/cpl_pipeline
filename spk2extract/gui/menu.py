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
    load_s2e_folder = QAction("Set s2e folder", parent)
    load_s2e_folder.triggered.connect(lambda: io.load_dialog_folder(parent))

    # load folder of processed data
    load_raw_smr = QAction("&Load raw .smr file", parent)
    load_raw_smr.setShortcut("Ctrl+L")
    load_raw_smr.triggered.connect(lambda: io.load_smr(parent))
    parent.addAction(load_raw_smr)

    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(load_raw_smr)


def run_s2e(parent):
    RW = rungui.RunWindow(parent)
    RW.show()
