import os
import pathlib
import sys
import warnings

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QComboBox
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QApplication,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QCheckBox,
    QLabel,
)

from spk2extract.gui import menu, traces
from spk2extract.gui.traces import MultiLine, QRangeSlider, DataPreparationThread


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.factor = 0
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.data_thread = None

        # -----------------------------------------------
        # set main window properties
        self.data = {}
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("spk2extract")
        import spk2extract

        s2e_dir = pathlib.Path(spk2extract.__file__).parent
        icon_path = os.fspath(s2e_dir.joinpath("logo", "logo.png"))
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))

        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: #0F111A; color: #8F93A2; border: 1px solid #4B526D;}")
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
        self.loaded = False

        # -----------------------------------------------
        # load user settings
        main_dir = pathlib.Path.home().joinpath(".clustersort")
        if not os.path.isdir(main_dir):
            # TODO: add warning that user_dir is being created to logs
            pass
        main_dir.mkdir(exist_ok=True)
        config_file = main_dir / "config.INI"

        menu.mainmenu(self)
        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        self.b0 = 0
        self.setAcceptDrops(True)
        self.show()

    def start_data_thread(self):
        if self.data_thread is not None:
            self.data_thread.quit()
            self.data_thread.wait()
    
        start_idx = int(self.range_slider.left_value)
        end_idx = int(self.range_slider.right_value)
        self.data_thread = DataPreparationThread(self.npy, start_idx, end_idx)
        self.data_thread.data_ready.connect(self.update_npy_plot)
        self.data_thread.start()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def make_graphics(self):
        ##### -------- MAIN PLOTTING AREA ---------- #####
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.scroll.setStyleSheet("QScrollArea {border: none;}")

        self.scroll_content = QWidget()
        self.vlayout = QVBoxLayout()

        self.plotWidgets = {}
        if self.loaded:
            # First, set up the plots
            for key in self.data.unit.keys():
                p = pg.PlotWidget()
                p.setMouseEnabled(x=True, y=False)
                p.enableAutoRange(x=True, y=True)
                p.setTitle(f"Plot for key: {key}")
                self.vlayout.addWidget(p)
                self.plotWidgets[key] = p

            self.scroll_content.setStyleSheet("background: #0F111A;")
            self.scroll_content.setLayout(self.vlayout)
            self.scroll.setWidget(self.scroll_content)

            self.l0.addWidget(self.scroll, 1, 2, self.b0 - 1, 30)
            traces.plot_multiple_traces(self)

    def make_graphics_npy(self):
        ##### -------- MAIN PLOTTING AREA ---------- #####
        self.downsample_box = QComboBox()
        self.downsample_box.addItem("0x")  # No downsampling
        for i in range(1, 7):
            self.downsample_box.addItem(f"{i * 10}x")
        self.downsample_box.currentIndexChanged.connect(self.update_downsample_factor)
        self.l0.addWidget(self.downsample_box, 0, 2, 1, 1)
        # self.vlayout.addWidget(self.downsample_box)

        self.scroll = QScrollArea()
        self.scroll.setStyleSheet("background-color: transparent; border: none;")
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        
        self.scroll_content = QWidget()
        self.vlayout = QVBoxLayout()

        self.plotWidgets = {}
        p = pg.PlotWidget(useOpenGL=True)
        p.setMouseEnabled(x=True, y=False)
        p.enableAutoRange(x=True, y=True)
        p.setTitle(f"Plot for npy")
        self.vlayout.addWidget(p)
        self.plotWidgets["npy"] = p
        self.scroll_content.setLayout(self.vlayout)
        self.scroll.setWidget(self.scroll_content)
        self.l0.addWidget(self.scroll, 1, 2, self.b0 - 1, 30)

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

    def update_npy_plot(self, x=None, sub_npy=None):
        p = self.plotWidgets["npy"]
        p.clear()

        start_idx = int(self.range_slider.left_value)
        end_idx = int(self.range_slider.right_value)

        # If start and end indices are not set, initialize to full range
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            start_idx = 0
            end_idx = self.npy.shape[0] - 1

        x = np.arange(self.npy.shape[1])

        if self.factor == 0:
            sub_npy = self.npy[start_idx:end_idx, :]
        else:
            sub_npy = self.npy[start_idx : end_idx : self.factor, :]

        multi_line = MultiLine(x, sub_npy)
        p.addItem(multi_line)
        p.autoRange()

    def update_downsample_factor(self):
        factor_str = self.downsample_box.currentText()
        self.factor = int(factor_str.rstrip('x'))
        self.update_npy_plot()

def run(statfile=None):
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    import spk2extract

    s2ppath = os.path.dirname(os.path.realpath(spk2extract.__file__))
    icon_path = os.path.join(s2ppath, "docs", "_static", "favicon.ico")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))

    app.setWindowIcon(app_icon)
    GUI = MainWindow()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    run()
