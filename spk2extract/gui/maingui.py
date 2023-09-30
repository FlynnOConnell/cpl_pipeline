import os, pathlib, shutil, sys, warnings

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QCheckBox, QLineEdit, QLabel

from . import menu, io, buttons, graphics
from ..defaults import defaults


class MainWindow(QMainWindow):

    def __init__(self, statfile=None):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("suite2p (run pipeline or load stat.npy)")
        import spk2extract
        s2p_dir = pathlib.Path(spk2extract.__file__).parent
        icon_path = os.fspath(s2p_dir.joinpath("logo", "logo.png"))

        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
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

        ### first time running, need to check for user files
        user_dir = pathlib.Path.home().joinpath(".suite2p")
        user_dir.mkdir(exist_ok=True)

        # check for classifier file
        class_dir = user_dir.joinpath("classifiers")
        class_dir.mkdir(exist_ok=True)

        # check for ops file (for running suite2p)
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
        self.rois = {"iROI": 0, "Sroi": 0, "Lam": 0, "LamMean": 0, "LamNorm": 0}
        self.colors = {"RGB": 0, "cols": 0, "colorbar": []}

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        b0 = self.make_buttons()
        self.make_graphics(b0)
        # so they"re on top of plot, draw last
        buttons.make_quadrants(self)

        # initialize merges
        self.merged = []
        self.imerge = [0]
        self.ichosen = 0
        self.rastermap = False

        # load initial file
        #statfile = "C:/Users/carse/OneDrive/Documents/suite2p/plane0/stat.npy"
        #statfile = "D:/grive/cshl_suite2p/GT1/suite2p/plane0/stat.npy"
        #statfile = "/media/carsen/DATA1/TIFFS/auditory_cortex/suite2p/plane0/stat.npy"
        #folder = "D:/DATA/GT1/singlechannel_half/suite2p/"
        #self.fname = folder
        #io.load_folder(self)
        if statfile is not None:
            self.fname = statfile
            io.load_proc(self)
            #self.manual_label()
        self.setAcceptDrops(True)
        self.show()
        self.win.show()

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

    def roi_text(self, state):
        if state == QtCore.Qt.Checked:
            for n in range(len(self.roi_text_labels)):
                if self.iscell[n] == 1:
                    self.p1.addItem(self.roi_text_labels[n])
                else:
                    self.p2.addItem(self.roi_text_labels[n])
            self.roitext = True
        else:
            for n in range(len(self.roi_text_labels)):
                if self.iscell[n] == 1:
                    try:
                        self.p1.removeItem(self.roi_text_labels[n])
                    except:
                        pass
                else:
                    try:
                        self.p2.removeItem(self.roi_text_labels[n])
                    except:
                        pass

            self.roitext = False

    def zoom_cell(self, state):
        if self.loaded:
            if state == QtCore.Qt.Checked:
                self.zoomtocell = True
            else:
                self.zoomtocell = False
            self.update_plot()

    def make_graphics(self, b0):
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        layout = self.win.ci.layout
        # --- cells image
        self.p1 = graphics.ViewBox(parent=self, lockAspect=True, name="plot1",
                                   border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p1, 0, 0)
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.view1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view1.autoDownsample = False
        self.color1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color1.autoDownsample = False
        self.p1.addItem(self.view1)
        self.p1.addItem(self.color1)
        self.view1.setLevels([0, 255])
        self.color1.setLevels([0, 255])
        #self.view1.setImage(np.random.rand(500,500,3))
        #x = np.arange(0,500)
        #img = np.concatenate((np.zeros((500,500,3)), 127*(1+np.tile(np.sin(x/100)[:,np.newaxis,np.newaxis],(1,500,1)))),axis=-1)
        #self.color1.setImage(img)
        # --- noncells image
        self.p2 = graphics.ViewBox(parent=self, lockAspect=True, name="plot2",
                                   border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p2, 0, 1)
        self.p2.setMenuEnabled(False)
        self.p2.scene().contextMenuItem = self.p2
        self.view2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view2.autoDownsample = False
        self.color2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color2.autoDownsample = False
        self.p2.addItem(self.view2)
        self.p2.addItem(self.color2)
        self.view2.setLevels([0, 255])
        self.color2.setLevels([0, 255])

        # LINK TWO VIEWS!
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")

        # --- fluorescence trace plot
        self.p3 = graphics.TraceBox(parent=self, invertY=False)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.enableAutoRange(x=True, y=True)
        self.win.addItem(self.p3, row=1, col=0, colspan=2)
        #self.p3 = pg.PlotItem()
        #self.v3.addItem(self.p3)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        layout = self.win.ci.layout
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)
        #self.win.scene().sigMouseClicked.connect(self.plot_clicked)

def run(statfile=None):
    # Always start by initializing Qt (only once per application)
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