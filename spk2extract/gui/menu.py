"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu
from pkg_resources import iter_entry_points

from . import io
from . import rungui
from spk2extract.spk_io.utils import get_spk2extract_path


def mainmenu(parent):
    main_menu = parent.menuBar()
    runS2E = QAction("&Run spk2extract", parent)
    runS2E.setShortcut("Ctrl+R")
    runS2E.triggered.connect(lambda: run_s2e(parent))
    parent.addAction(runS2E)

    # load processed data
    loadProc = QAction("&Load processed data", parent)
    loadProc.setShortcut("Ctrl+L")
    loadProc.triggered.connect(lambda: io.load_dialog(parent))
    parent.addAction(loadProc)

    # load folder of processed data
    loadFolder = QAction("Load &Folder with planeX folders", parent)
    loadFolder.setShortcut("Ctrl+F")
    loadFolder.triggered.connect(lambda: io.load_dialog_folder(parent))
    parent.addAction(loadFolder)

    # export figure
    exportFig = QAction("Export as image (svg)", parent)
    exportFig.triggered.connect(lambda: io.export_fig(parent))
    exportFig.setEnabled(True)
    parent.addAction(exportFig)

    # export figure
    parent.manual = QAction("Manual labelling", parent)
    parent.manual.setEnabled(False)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(runS2E)
    file_menu.addAction(loadProc)
    file_menu.addAction(loadFolder)
    file_menu.addAction(exportFig)
    file_menu.addAction(parent.manual)


def run_s2e(parent):
    RW = rungui.RunWindow(parent)
    RW.show()
