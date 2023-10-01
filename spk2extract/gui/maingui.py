import os, pathlib, shutil, sys, warnings

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QCheckBox, QLineEdit, QLabel

from spk2extract.gui import menu, io, buttons, graphics
from ..defaults import defaults


class MainWindow(QMainWindow):

    def __init__(self, statfile=None):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("spk2extract (run pipeline or load stat.npy)")
        import spk2extract
        s2e_dir = pathlib.Path(spk2extract.__file__).parent
        icon_path = os.fspath(s2e_dir.joinpath("logo", "logo.png"))

        app_icon = QtGui.QIcon()

        app_icon.addFile(icon_path, QtCore.QSize(32, 32))

        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.loaded = False
        self.ops_plot = []

        user_dir = pathlib.Path.home().joinpath(".spk2extract")
        user_dir.mkdir(exist_ok=True)

        # check for ops file (for running spk2extract)
        ops_dir = user_dir.joinpath("ops")
        ops_dir.mkdir(exist_ok=True)
        self.opsuser = os.fspath(ops_dir.joinpath("ops_user.npy"))
        if not os.path.isfile(self.opsuser):
            np.save(self.opsuser, defaults())
        self.opsfile = self.opsuser

        menu.mainmenu(self)

        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)

        # default plot options
        self.ops_plot = {
            "ROIs_on": True,
            "color": 0,
            "view": 0,
            "opacity": [127, 255],
            "saturation": [0, 255],
            "colormap": "hsv"
        }
        self.colors = {"RGB": 0, "cols": 0, "colorbar": []}

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        b0 = self.make_buttons()

        # initialize merges
        self.merged = []
        self.imerge = [0]
        self.ichosen = 0
        self.rastermap = False

        # load initial file
        #statfile = "C:/Users/carse/OneDrive/Documents/spk2extract/plane0/stat.npy"
        #statfile = "D:/grive/cshl_spk2extract/GT1/spk2extract/plane0/stat.npy"
        #statfile = "/media/carsen/DATA1/TIFFS/auditory_cortex/spk2extract/plane0/stat.npy"
        #folder = "D:/DATA/GT1/singlechannel_half/spk2extract/"
        #self.fname = folder
        #spk_io.load_folder(self)
        if statfile is not None:
            self.fname = statfile
            io.load_proc(self)
            #self.manual_label()
        self.setAcceptDrops(True)
        self.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        print(files)
        self.fname = files[0]
        if os.path.splitext(self.fname)[-1] == ".npy":
            io.load_proc(self)
        elif os.path.splitext(self.fname)[-1] == ".nwb":
            io.load_NWB(self)
        else:
            print("invalid extension %s, use .nwb or .npy" %
                  os.path.splitext(self.fname)[-1])

    def make_buttons(self):
        # ROI CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBox = QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.toggle()
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)


def run(statfile=None):
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    import spk2extract
    s2ppath = os.path.dirname(os.path.realpath(spk2extract.__file__))
    icon_path = os.path.join(s2ppath, "docs", "_static", "favicon.ico")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))

    app.setWindowIcon(app_icon)
    GUI = MainWindow(statfile=statfile)
    ret = app.exec_()
    sys.exit(ret)

if __name__ == "__main__":
    run()