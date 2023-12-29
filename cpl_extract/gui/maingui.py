import os
import pathlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import vispy
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QApplication,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QLabel,
    QProgressBar,
)
from PyQt5.QtWidgets import (
    QSlider,
    QComboBox,
    QPushButton,
)
from icecream import ic

from cpl_extract import load_pickled_object
from cpl_extract.gui import menu, io
from cpl_extract.gui.traces import (
    QRangeSlider,
    WaveformPlot,
    PlotDialog,
    ClusterSelectors,
    DataLoader, PROCESSING_STEPS,
)
from cpl_extract.gui.widgets import MultiSelectionDialog


def set_deep_ocean_theme(app):
    palette = QPalette()

    # Set color roles from your Deep Ocean color scheme
    palette.setColor(QPalette.Window, QColor("#0F111A"))
    palette.setColor(QPalette.WindowText, QColor("#8F93A2"))
    palette.setColor(QPalette.Base, QColor("#181A1F"))
    palette.setColor(QPalette.ToolTipBase, QColor("#181A1F"))
    palette.setColor(QPalette.ToolTipText, QColor("#8F93A2"))
    palette.setColor(QPalette.AlternateBase, QColor("#191A21"))
    palette.setColor(QPalette.Text, QColor("#4B526D"))
    palette.setColor(QPalette.Button, QColor("#191A21"))
    palette.setColor(QPalette.ButtonText, QColor("#8F93A2"))
    palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Highlight, QColor("#1F2233"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#464B5D"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#464B5D"))

    app.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.npy = None
        self.factor = 0
        self.data_thread = None
        self.data = {}
        self.loaded = False
        self.plotWidgets = {}
        self.base_path = None
        self.files = []
        self.channels = []
        self.clusters = []
        self.dataset = None
        self.h5_file = None
        self.h5_file_path = None
        self.raw_file_type = None

        io.select_base_folder(self)
        if self.base_path is None:
            raise ValueError("Must have a valid base directory")
        else:
            self.base_path = Path(self.base_path)


        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setWindowTitle("cpl_extract")
        self.setGeometry(50, 50, 1500, 800)

        self.mainLayout = QVBoxLayout()
        top_area = QWidget()
        self.top_layout = QVBoxLayout()

        top_area.setLayout(self.top_layout)
        top_area.setFixedHeight(80)

        # Add top area to main layout
        self.mainLayout.addWidget(top_area)
        self.open_area = QWidget()
        self.open_layout = QGridLayout()
        self.open_area.setLayout(self.open_layout)

        # ------------- Bottom Area for Further Items ----------------
        self.bottom_area = QWidget()
        self.bottom_layout = QGridLayout()
        self.bottom_area.setLayout(self.bottom_layout)

        # Add bottom area to main layout
        self.mainLayout.addWidget(self.open_area)
        self.mainLayout.setStretch(1, 1)  # Makes the bottom area take up the remaining space

        # Set main layout to central widget
        cwidget = QWidget()
        cwidget.setLayout(self.mainLayout)
        self.setCentralWidget(cwidget)
        menu.mainmenu(self)

        self.setAcceptDrops(True)
        self.data_thread = DataLoader(str(self.base_path))
        self.data_thread.dataLoaded.connect(self.display_processing_directory)
        self.show()

    def start_layout(self):
        layout = QGridLayout()
        self.start_button = QPushButton("Start")
        # self.start_button.clicked.connect()
        self.start_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        layout.addWidget(self.start_button, 0, 0)
        layout.addWidget(self.progress, 0, 1)
        self.top_layout.addLayout(layout, 0, 0)

    def handle_process(self, selection):
        if isinstance(selection, list):
            selection = selection[0]
        selection = Path(selection)
        if selection.with_suffix('.p'):
            load_pickled_object(selection)
            ic()

    def display_processing_directory(self,):
        """Opens a dialog to select multiple clusters in the center of the screen, resizable"""
        pkl_file = self.data_thread.get_pk()
        if pkl_file:
            pickle_files = list(self.base_path.glob('**/*.p'))
            pk = io.load_pickled_object(pickle_files[0])
            self.data = pk
        else:
            print('no pickle file found')


    def draw_plots(self):
        plot_dialog = PlotDialog(self.plotWidgets)
        plot_dialog.exec_()

    def clear_plots(self):
        for key in self.plotWidgets.keys():
            self.bottom_layout.removeWidget(self.plotWidgets[key])
        self.plotWidgets = {}

    def get_channels(self):
        selected_file = self.file_selector.currentText()
        if not selected_file:
            return []
        return sorted(self.data[selected_file].keys())

    def get_clusters(self):
        selected_file = self.file_selector.currentText()
        selected_channel = self.channel_selector.currentText()
        if not selected_file or not selected_channel:
            return []
        # To get only the 'n_clusters' and not the .npy file paths
        all_keys = self.data[selected_file][selected_channel].keys()
        cluster_keys = [k for k in all_keys if "_clusters" in k]
        return sorted(cluster_keys)

    def get_single_clusters(self):
        selected_file = self.file_selector.currentText()
        selected_channel = self.channel_selector.currentText()
        selected_cluster = self.cluster_selector.currentText()
        if not selected_file or not selected_channel or not selected_cluster:
            return []
        return sorted(
            self.data[selected_file][selected_channel][selected_cluster].keys()
        )

    def get_current_channel_data(self):
        selected_file = self.file_selector.currentText()
        selected_channel = self.channel_selector.currentText()
        if not selected_file or not selected_channel:
            return []
        return self.data[selected_file][selected_channel]

    def get_vispy(self, data=None, title="") -> WaveformPlot:
        vispy.use("PyQt5")
        return WaveformPlot(self, data, plot_title=title)

    def pull_channel_data(self):
        cluster_folder_path = (
            self.base_path
            / self.file_selector.currentText()
            / "Data"
            / self.channel_selector.currentText()
            / self.cluster_selector.currentText()
            / self.single_cluster_selector.currentText()
        )
        if os.path.exists(cluster_folder_path):
            for file in os.listdir(cluster_folder_path):
                if file.endswith(".npy") and file.startswith("cluster_spikes"):
                    self.npy = np.load(os.path.join(cluster_folder_path, file))
                    self.loaded = True
                    break


def run():
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)

    path = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(path, "docs", "_static", "favicon.png")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))

    set_deep_ocean_theme(app)

    app.setWindowIcon(app_icon)
    MainWindow()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
   cache = Path().home() / '.cache'
   cache.mkdir(exist_ok=True)
   os.environ['CACHE_DIR'] = str(cache)
   run()
