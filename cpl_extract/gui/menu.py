"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from PyQt5.QtWidgets import QMenu, QAction

from . import io
from . import rungui


def mainmenu(parent):
    main_menu = parent.menuBar()
    load_s2e_folder = QAction("Set base folder", parent)
    load_s2e_folder.setShortcut("Ctrl+B")
    load_s2e_folder.triggered.connect(lambda: io.select_base_folder(parent))

    load_npy = QAction("Load .npy data", parent)
    load_npy.setShortcut("Ctrl+L")
    load_npy.triggered.connect(lambda: io.load_npy(parent))

    # load folder of processed data
    debug = QAction("Debug", parent)
    debug.triggered.connect(lambda: parent.debug())

    file_menu = main_menu.addMenu("&File")
    main_menu.addAction(debug)
    load_menu = file_menu.addMenu("&Load")
    load_menu.addAction(load_s2e_folder)


def run_s2e(parent):
    RW = rungui.RunWindow(parent)
    RW.show()
