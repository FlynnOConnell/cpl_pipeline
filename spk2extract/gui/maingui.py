import os
import pathlib
import sys
import warnings

import numpy as np
import pyqtgraph as pg
import vispy
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread
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

from spk2extract.gui import menu
from spk2extract.gui.traces import (
    QRangeSlider,
    WaveformPlot,
    PlotDialog,
    ClusterSelectors,
    DataLoader,
)
from spk2extract.gui.widgets import MultiSelectionDialog


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
        self.factor = 0
        self.data_thread = None
        self.data = {}
        self.loaded = False
        self.plotWidgets = {}

        # Load data directory container
        self.base_path = pathlib.Path().home() / "clustersort"
        self.data_thread = DataLoader(str(self.base_path))
        self.data_thread.dataLoaded.connect(self.update_data)

        self.loading_label = QLabel("Loading...")
        self.loading_spinner = QProgressBar()
        self.loading_spinner.setRange(0, 0)  # Indeterminate progress

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setWindowTitle("spk2extract")
        self.setGeometry(50, 50, 1500, 800)

        self.mainLayout = QVBoxLayout()
        top_area = QWidget()
        self.top_layout = QGridLayout()

        # Initially add loading widgets
        self.top_layout.addWidget(self.loading_label, 0, 0)
        self.top_layout.addWidget(self.loading_spinner, 0, 1)

        self.file_selector = ClusterSelectors(self.get_files)
        self.channel_selector = ClusterSelectors(self.get_channels)
        self.cluster_selector = ClusterSelectors(self.get_clusters)
        self.single_cluster_selector = MultiSelectionDialog(self.get_single_clusters)
        self.single_cluster_selector_label = QPushButton("Select Clusters")
        self.single_cluster_selector_label.clicked.connect(
            self.open_multi_select_dialog
        )

        self.file_selector.data_changed.connect(self.channel_selector.populate)
        self.file_selector.data_changed.connect(self.cluster_selector.populate)
        self.file_selector.data_changed.connect(self.single_cluster_selector.populate)

        self.channel_selector.data_changed.connect(self.cluster_selector.populate)
        self.channel_selector.data_changed.connect(
            self.single_cluster_selector.populate
        )

        self.cluster_selector.data_changed.connect(
            self.single_cluster_selector.populate
        )

        self.file_selector.populate()
        self.channel_selector.populate()
        self.cluster_selector.populate()
        self.single_cluster_selector.populate()

        self.vispy_button = QPushButton("View")
        self.vispy_button.clicked.connect(self.draw_plots)
        self.vispy_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        self.clear_button = QPushButton("Clear")

        # Set stretch factors
        self.top_layout.setColumnStretch(0, 0)
        self.top_layout.setColumnStretch(1, 0)
        self.top_layout.setColumnStretch(2, 0)
        self.top_layout.setColumnStretch(3, 0)

        top_area.setLayout(self.top_layout)
        top_area.setFixedHeight(80)

        # Add top area to main layout
        self.mainLayout.addWidget(top_area)

        # ------------- Bottom Area for Further Items ----------------
        self.bottom_area = QWidget()
        self.bottom_layout = QGridLayout()
        self.bottom_area.setLayout(self.bottom_layout)

        # Add bottom area to main layout
        self.mainLayout.addWidget(self.bottom_area)
        self.mainLayout.setStretch(
            1, 1
        )  # Makes the bottom area take up the remaining space

        # Set main layout to central widget
        cwidget = QWidget()
        cwidget.setLayout(self.mainLayout)
        self.setCentralWidget(cwidget)
        menu.mainmenu(self)
        self.setAcceptDrops(True)
        self.show()

    def debug(self):
        x = 5
        y = self.base_path

    def get_current_channel_path(self):
        return (
            self.base_path
            / self.file_selector.currentText()
            / "Data"
            / self.channel_selector.currentText()
        )

    def get_current_cluster_group_path(self):
        return self.get_current_channel_path() / self.cluster_selector.currentText()

    def update_data(self, data):
        self.top_layout.removeWidget(self.loading_label)
        self.top_layout.removeWidget(self.loading_spinner)
        self.loading_label.deleteLater()
        self.loading_spinner.deleteLater()

        self.top_layout.addWidget(self.file_selector, 0, 0)
        self.top_layout.addWidget(self.channel_selector, 0, 1)
        self.top_layout.addWidget(self.cluster_selector, 0, 2)
        self.top_layout.addWidget(self.single_cluster_selector_label, 0, 3)
        self.top_layout.addWidget(self.vispy_button, 0, 4)

        self.data = data
        self.file_selector.populate()

    def handle_single_cluster_selection(self, selected_clusters):
        paths = [
            self.get_current_cluster_group_path() / cluster
            for cluster in selected_clusters
        ]
        for path in paths:
            cluster_npy = path / "cluster_spikes.npy"
            data = np.load(cluster_npy)
            self.plotWidgets[path.name] = self.get_vispy(data, path.name)

    def open_multi_select_dialog(
        self,
    ):
        dialog = MultiSelectionDialog(self.get_single_clusters)
        dialog.selection_made.connect(self.handle_single_cluster_selection)
        dialog.resize(200, min(400, len(self.get_single_clusters()) * 20 + 50))
        dialog.move(self.frameGeometry().center() - dialog.rect().center())
        dialog.exec_()

    def draw_plots(self):
        plot_dialog = PlotDialog(self.plotWidgets)
        plot_dialog.exec_()

    def clear_plots(self):
        for key in self.plotWidgets.keys():
            self.bottom_layout.removeWidget(self.plotWidgets[key])
        self.plotWidgets = {}

    def get_files(self):
        return sorted(self.data.keys())

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
        cluster_path = self.selected_cluster_path
        if os.path.exists(cluster_folder_path):
            for file in os.listdir(cluster_folder_path):
                if file.endswith(".npy") and file.startswith("cluster_spikes"):
                    self.npy = np.load(os.path.join(cluster_folder_path, file))
                    self.loaded = True
                    break

    @staticmethod
    def get_npy(self, path):
        return np.load(path)

    def make_graphics_npy(self):
        self.downsample_box = QComboBox()
        self.downsample_box.addItem("0x")
        for i in range(1, 7):
            self.downsample_box.addItem(f"{i * 10}x")
        self.downsample_box.currentIndexChanged.connect(self.update_downsample_factor)
        self.downsample_box.setToolTip("Downsample factor")
        self.top_layout.addWidget(self.downsample_box, 0, 2, 1, 1)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.vlayout = QVBoxLayout()

        self.plot = pg.PlotWidget(
            useOpenGL=True,
        )
        p = self.plot
        p.setMouseEnabled(x=False, y=True)
        p.enableAutoRange(x=False, y=True)
        p.setTitle(f"Plot for npy")

        self.vlayout.addWidget(p)
        self.plotWidgets["npy"] = p
        self.scroll_content.setLayout(self.vlayout)
        self.scroll.setWidget(self.scroll_content)
        self.top_layout.addWidget(self.downsample_box, 0, 4)
        self.bottom_layout.addWidget(self.scroll, 0, 0, 1, -1)

        # Custom range slider in traces.py
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setMinimum(0)
        self.start_slider.setMaximum(self.npy.shape[0] - 1)
        self.start_slider.setValue(0)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setMinimum(0)
        self.end_slider.setMaximum(self.npy.shape[0] - 1)
        self.end_slider.setValue(self.npy.shape[0] - 1)

        self.range_slider = QRangeSlider(min_val=0, max_val=self.npy.shape[0] - 1)
        self.range_slider.setMinimumHeight(50)

        self.range_slider.left_value = 0
        self.range_slider.right_value = self.npy.shape[0] - 1
        self.range_slider.range_changed.connect(self.start_data_thread)
        self.vlayout.addWidget(self.range_slider)
        self.vlayout.addStretch()
        self.update_npy_plot()


def run():
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    import spk2extract

    s2ppath = os.path.dirname(os.path.realpath(spk2extract.__file__))
    icon_path = os.path.join(s2ppath, "docs", "_static", "favicon.ico")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))

    set_deep_ocean_theme(app)

    app.setWindowIcon(app_icon)
    MainWindow()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    run()
