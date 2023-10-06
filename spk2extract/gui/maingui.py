import os
import pathlib
import sys
import warnings

import numpy as np
import pyqtgraph as pg
import vispy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QSlider,
    QComboBox,
    QToolBar,
    QSplitter,
    QTextEdit,
    QPushButton,
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QApplication,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)
from vispy import scene
from vispy.scene import visuals

from spk2extract.gui import menu, traces
from spk2extract.gui.traces import MultiLine, QRangeSlider, DataPreparationThread, VispyCanvas


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
        from pathlib import Path

        # Flags
        self.__file_selector_populated = False
        self.__channel_selector_populated = False
        self.__cluster_selector_populated = False
        self.__single_cluster_selector_populated = False

        self.__updating_file_selector = False
        self.__updating_channel_selector = False
        self.__updating_cluster_selector = False
        self.__updating_single_cluster_selector = False

        # Prepopulating vars
        self.factor = 0
        self.data_thread = None
        self.data = {}
        self.loaded = False
        self.plotWidgets = {}
        self.base_path = Path().home() / "clustersort"

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setWindowTitle("spk2extract")
        self.setGeometry(50, 50, 1500, 800)
        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        self.stylePressed = (
            "QPushButton {Text-align: left; "
            "background-color: rgb(100,50,100); "
            "color:white;}"
        )
        self.styleUnpressed = (
            "QPushButton {Text-align: left; "
            "background-color: rgb(50,50,50); "
            "color:white;}"
        )
        self.styleInactive = (
            "QPushButton {Text-align: left; "
            "background-color: rgb(50,50,50); "
            "color:gray;}"
        )

        main_dir = pathlib.Path.home().joinpath("clustersort")
        if not os.path.isdir(main_dir):
            # TODO: add warning that user_dir is being created to logs
            pass
        main_dir.mkdir(exist_ok=True)

        # --------- MAIN WIDGET LAYOUT ---------------------
        # Create main layout as QVBoxLayout
        self.mainLayout = QVBoxLayout()

        top_area = QWidget()
        self.top_layout = QGridLayout()

        # Create ComboBox and Button for top area
        self.file_selector = QComboBox()
        self.file_selector.setToolTip("Select a file")
        self.file_selector.currentIndexChanged.connect(self.update_file_selector)
        self.file_selector.currentIndexChanged.connect(self.on_file_selected)

        self.channel_selector = QComboBox()
        self.channel_selector.setToolTip("Select a channel")
        self.channel_selector.currentIndexChanged.connect(self.update_channel_selector)
        self.channel_selector.currentIndexChanged.connect(self.on_channel_selected)

        self.cluster_selector = QComboBox()
        self.cluster_selector.setToolTip("Select a cluster")
        self.cluster_selector.currentIndexChanged.connect(self.update_cluster_selector)
        self.cluster_selector.currentIndexChanged.connect(self.on_cluster_selected)

        self.single_cluster_selector = QComboBox()
        self.single_cluster_selector.setToolTip("Select a single cluster")
        self.single_cluster_selector.currentIndexChanged.connect(
            self.update_single_cluster_selector
        )
        self.single_cluster_selector.activated.connect(
            self.reupdate_single_cluster_selector
        )

        self.plot_type_selector = QComboBox()
        self.plot_type_selector.setToolTip("Type")
        self.plot_type_selector.currentIndexChanged.connect(self.plot_type)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setToolTip("Analyze selected file and channel")
        self.analyze_button.clicked.connect(self.pull_channel_data)

        self.vispy_button = QPushButton("Vispy")
        self.vispy_button.setToolTip("Run Vispy")
        self.vispy_button.clicked.connect(self.plot_vispy)

        # Add widgets to top_layout
        self.top_layout.addWidget(self.file_selector, 0, 0)
        self.top_layout.addWidget(self.channel_selector, 0, 1)
        # self.top_layout.addWidget(self.plot_type_selector, 0, 2)
        self.top_layout.addWidget(self.cluster_selector, 0, 2)
        self.top_layout.addWidget(self.single_cluster_selector, 0, 3)
        self.top_layout.addWidget(self.analyze_button, 0, 4)
        self.top_layout.addWidget(self.vispy_button, 0, 5)

        # Set stretch factors
        self.top_layout.setColumnStretch(0, 1)
        self.top_layout.setColumnStretch(1, 1)
        self.top_layout.setColumnStretch(2, 1)
        self.top_layout.setColumnStretch(3, 1)

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
        self.update_file_selector()
        self.update_channel_selector()
        self.update_cluster_selector()
        self.update_single_cluster_selector()
        self.show()

    def update_plot(self, event):
        pass

    def plot_vispy(self):
        vispy.use("PyQt5")
        self.pull_channel_data()
        canvas = VispyCanvas(self)
        canvas.plot()
        canvas.run()

    def plot_type(self):
        types = ["raw", "clusters", "isi"]
        for t in types:
            pass
        return types[self.plot_type_selector.currentIndex()]

    def update_file_selector(self):
        if self.__updating_file_selector:
            return

        self.__updating_file_selector = True

        if not self.__file_selector_populated:
            self.file_selector.clear()

            if not self.base_path:
                self.file_selector.addItem("Please set a base path")
            else:
                self.file_selector.addItem("Select a file")
                for foldername in sorted(os.listdir(self.base_path)):
                    folder_path = os.path.join(self.base_path, foldername)
                    if os.path.isdir(folder_path):
                        self.file_selector.addItem(foldername)
            self.__file_selector_populated = True
        self.__updating_file_selector = False

    def on_file_selected(self):
        """When a file is selected, update the channel selector"""
        selected_file = self.file_selector.currentText()
        if selected_file != "Select a file":
            self.update_channel_selector()
            self.update_cluster_selector()
            self.update_single_cluster_selector()

    def on_channel_selected(self):
        """When a channel is selected, update the cluster selector"""
        selected_file = self.file_selector.currentText()
        selected_channel = self.channel_selector.currentText()
        if selected_file and selected_channel != "Select a file":
            self.update_cluster_selector()
            self.update_single_cluster_selector()

    def on_cluster_selected(self):
        """When a channel is selected, update the cluster selector"""
        selected_file = self.file_selector.currentText()
        selected_channel = self.channel_selector.currentText()
        selected_cluster = self.cluster_selector.currentText()
        if selected_file and selected_channel and selected_cluster != "Select a file":
            self.update_single_cluster_selector()

    def update_channel_selector(self):
        if self.__channel_selector_populated:
            return
        if self.__updating_channel_selector:
            return

        self.__updating_channel_selector = True
        self.channel_selector.clear()

        # If no file is chosen, populate with "Select a file"
        if (
            not self.file_selector.currentText()
            or self.file_selector.currentText() == "Select a file"
        ):
            self.channel_selector.addItem("Select a file")
        else:
            self.channel_selector.addItem("Select a channel")
            data_folder_path = (
                self.base_path / self.file_selector.currentText() / "Data"
            )
            if data_folder_path.is_dir():
                for channel in sorted(os.listdir(data_folder_path)):
                    channel_path = data_folder_path / channel
                    if channel_path.is_dir():
                        self.channel_selector.addItem(channel)
                self.__channel_selector_populated = True
        self.__updating_channel_selector = False

    def update_cluster_selector(self):
        if self.__cluster_selector_populated:
            return
        if self.__updating_cluster_selector:
            return

        self.__updating_cluster_selector = True
        self.cluster_selector.clear()

        # No file chosen, "select a file"
        if (
            not self.file_selector.currentText()
            or self.file_selector.currentText() == "Select a file"
        ):
            self.cluster_selector.addItem("Select a file")

        # No channel chosen, "select a channel"
        elif (
            self.channel_selector.currentText()
            == "Select a channel"  # this doesn't occur when a file is chosen, it's still ''
            or not self.channel_selector.currentText()
        ):
            self.cluster_selector.clear()
            self.cluster_selector.addItem("Select a channel")
        else:
            self.cluster_selector.clear()
            self.cluster_selector.addItem("Select a cluster")
            channel_folder_path = (
                self.base_path
                / self.file_selector.currentText()
                / "Data"
                / self.channel_selector.currentText()
            )
            if channel_folder_path.is_dir():
                for cluster in sorted(os.listdir(channel_folder_path)):
                    channel_path = channel_folder_path / cluster
                    if channel_path.is_dir():
                        self.cluster_selector.addItem(cluster)
                self.__cluster_selector_populated = True
        self.__updating_cluster_selector = False

    def update_single_cluster_selector(self):
        if self.__single_cluster_selector_populated:
            return
        if self.__updating_single_cluster_selector:
            return

        self.__updating_single_cluster_selector = True
        self.single_cluster_selector.clear()

        # No file chosen, "select a file"
        if (
            not self.file_selector.currentText()
            or self.file_selector.currentText() == "Select a file"
        ):
            self.single_cluster_selector.addItem("Select a file")

        # No channel chosen, "select a channel"
        elif (
            self.channel_selector.currentText()
            == "Select a channel"  # this doesn't occur when a file is chosen, it's still ''
            or not self.channel_selector.currentText()
        ):
            self.single_cluster_selector.clear()
            self.single_cluster_selector.addItem("Select a channel")

        # No cluster chosen, "select a cluster"
        elif (
            self.cluster_selector.currentText()
            == "Select a cluster"  # this doesn't occur when a file is chosen, it's still ''
            or not self.cluster_selector.currentText()
        ):
            self.single_cluster_selector.clear()
            self.single_cluster_selector.addItem("Select a cluster")
        else:
            self.single_cluster_selector.addItem("Select a single cluster")
            cluster_folder_path = (
                self.base_path
                / self.file_selector.currentText()
                / "Data"
                / self.channel_selector.currentText()
                / self.cluster_selector.currentText()
            )
            if cluster_folder_path.is_dir():
                for this_cluster in sorted(os.listdir(cluster_folder_path)):
                    channel_path = cluster_folder_path / this_cluster
                    if channel_path.is_dir():
                        self.single_cluster_selector.addItem(this_cluster)
                self.__single_cluster_selector_populated = True
        self.__updating_single_cluster_selector = False

    def reupdate_single_cluster_selector(self):
        self.single_cluster_selector.clear()
        cluster_folder_path = (
            self.base_path
            / self.file_selector.currentText()
            / "Data"
            / self.channel_selector.currentText()
            / self.cluster_selector.currentText()
        )
        if cluster_folder_path.is_dir():
            for this_cluster in sorted(os.listdir(cluster_folder_path)):
                channel_path = cluster_folder_path / this_cluster
                if channel_path.is_dir():
                    self.single_cluster_selector.addItem(this_cluster)

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
                    self.make_graphics_npy()
                    self.start_data_thread()
                    break

    def start_data_thread(self):
        if self.data_thread is not None:
            self.data_thread.quit()
            self.data_thread.wait()

        start_idx = int(self.range_slider.left_value)
        end_idx = int(self.range_slider.right_value)
        self.data_thread = DataPreparationThread(self.npy, start_idx, end_idx)
        self.data_thread.data_ready.connect(self.update_npy_plot)
        self.data_thread.start()

    def make_graphics(self):
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.vlayout = QVBoxLayout()

        if self.loaded:
            # First, set up the plots
            for key in self.data.unit.keys():
                p = pg.PlotWidget()
                p.setMouseEnabled(x=True, y=False)
                p.enableAutoRange(x=True, y=True)
                p.setTitle(f"Plot for key: {key}")
                self.vlayout.addWidget(p)
                self.plotWidgets[key] = p

            self.scroll_content.setLayout(self.vlayout)
            self.scroll.setWidget(self.scroll_content)

            self.layout_plot_info.addWidget(self.scroll, 1, 2, self.b0 - 1, 30)
            traces.plot_multiple_traces(self)

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

        self.plot = pg.PlotWidget(useOpenGL=True,)
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

    def update_npy_plot(
        self,
    ):
        p = self.plotWidgets["npy"]
        p.clear()

        start_idx = int(self.range_slider.left_value)
        end_idx = int(self.range_slider.right_value)

        # If start and end indices are not set, initialize to full range
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            start_idx = 0
            end_idx = self.npy.shape[0] - 1

        x = np.arange(self.npy.shape[1])
        sub_npy = self.npy[start_idx:end_idx, :]

        multi_line = MultiLine(x, sub_npy)
        p.addItem(multi_line)
        p.autoRange()

    def update_downsample_factor(self):
        factor_str = self.downsample_box.currentText()
        self.factor = int(factor_str.rstrip("x"))
        p = self.plotWidgets["npy"]
        p.clear()
        # Redraw plot with new downsampling factor
        # Avoiding ValueError: slice step cannot be zero
        if self.factor != 0:
            p.addItem(
                MultiLine(np.arange(self.npy.shape[1]), self.npy[:: self.factor, :])
            )
        else:
            p.addItem(MultiLine(np.arange(self.npy.shape[1]), self.npy))


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
