import os
import pathlib
import sys
import warnings

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QSlider, QComboBox
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QApplication,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)

from spk2extract.gui import menu, traces
from spk2extract.gui.traces import MultiLine, QRangeSlider, DataPreparationThread


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
        pg.setConfigOptions(imageAxisOrder="row-major")

        # -----------------------------------------------
        # set main window properties
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

        # -----------------------------------------------
        # load user settings

        main_dir = pathlib.Path.home().joinpath(".clustersort")
        if not os.path.isdir(main_dir):
            # TODO: add warning that user_dir is being created to logs
            pass
        main_dir.mkdir(exist_ok=True)

        # --------- MAIN WIDGET LAYOUT ---------------------
        menu.mainmenu(self)
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

            self.l0.addWidget(self.scroll, 1, 2, self.b0 - 1, 30)
            traces.plot_multiple_traces(self)

    def make_graphics_npy(self):

        self.downsample_box = QComboBox()
        self.downsample_box.addItem("0x")  # No downsampling
        for i in range(1, 7):
            self.downsample_box.addItem(f"{i * 10}x")
        self.downsample_box.currentIndexChanged.connect(self.update_downsample_factor)
        self.downsample_box.setToolTip("Downsample factor")
        self.l0.addWidget(self.downsample_box, 0, 2, 1, 1)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        
        self.scroll_content = QWidget()
        self.vlayout = QVBoxLayout()

        self.plot = pg.PlotWidget(useOpenGL=True)
        p = self.plot
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
        sub_npy = self.npy[start_idx:end_idx, :]

        multi_line = MultiLine(x, sub_npy)
        p.addItem(multi_line)
        p.autoRange()

    def update_downsample_factor(self):
        factor_str = self.downsample_box.currentText()
        self.factor = int(factor_str.rstrip('x'))
        p = self.plotWidgets["npy"]
        p.clear()
        # Redraw plot with new downsampling factor
        # Avoiding ValueError: slice step cannot be zero
        if self.factor != 0:
            p.addItem(MultiLine(np.arange(self.npy.shape[1]), self.npy[::self.factor, :]))
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
