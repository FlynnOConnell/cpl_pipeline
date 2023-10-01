import os
import pathlib
import sys
import warnings

import pyqtgraph as pg
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

from spk2extract.gui import menu, io, buttons, traces, views


class MainWindow(QMainWindow):
    def __init__(self, statfile=None):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

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
        self.setStyleSheet("QMainWindow {background: 'black';}")
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
        self.ops_plot = []

        # -----------------------------------------------
        # load user settings
        main_dir = pathlib.Path.home().joinpath(".clustersort")
        if not os.path.isdir(main_dir):
            # TODO: add warning that user_dir is being created to logs
            pass
        main_dir.mkdir(exist_ok=True)
        config_file = main_dir / "config.INI"
        if not os.path.isfile(config_file):
            # shutil.copyfile(s2e_dir.joinpath("defaults", "config.INI"), config_file)
            pass

        menu.mainmenu(self)
        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        self.ops_plot = {
            "ROIs_on": True,
            "color": 0,
            "view": 0,
            "opacity": [127, 255],
            "saturation": [0, 255],
            "colormap": "hsv",
        }
        self.colors = {"RGB": 0, "cols": 0, "colorbar": []}

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        self.b0 = self.make_buttons()
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
            print(
                "invalid extension %s, use .nwb or .npy"
                % os.path.splitext(self.fname)[-1]
            )

    def make_buttons(self) -> int:
        """
        Make buttons for the main window.

        Returns
        -------
        b0 : int
            A running count of the number of buttons added to the window, one per row

        """
        # ROI CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBox = QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.toggle()
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)

        buttons.make_selection(self)
        b0 = views.make_buttons(self)  # b0 says how many
        b0 += 1

        self.stats_to_show = [
            "med",
            "npix",
            "skew",
            "compact",
            "footprint",
            "aspect_ratio",
        ]
        lilfont = QtGui.QFont("Arial", 8)
        qlabel = QLabel(self)
        qlabel.setFont(self.boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, b0, 0, 1, 1)
        b0 += 1
        self.l0.addWidget(QLabel(""), b0, 0, 1, 2)
        self.l0.setRowStretch(b0, 1)
        b0 += 2
        b0 = traces.make_buttons(self, b0)
        return b0

    def make_graphics(self):
        ##### -------- MAIN PLOTTING AREA ---------- #####
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.scroll.setStyleSheet("QScrollArea {background: #0F111A; border: none;}")

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
                p.setStyleSheet("background: #0F111A; color: #8F93A2; border: 1px solid #4B526D;")

                self.vlayout.addWidget(p)
                self.plotWidgets[key] = p

            self.scroll_content.setStyleSheet("background: #0F111A;")
            self.scroll_content.setLayout(self.vlayout)
            self.scroll.setWidget(self.scroll_content)

            self.l0.addWidget(self.scroll, 1, 2, self.b0 - 1, 30)
            traces.plot_multiple_traces(self)

    def keyPressEvent(self, event):
        if self.loaded:
            if (
                event.modifiers() != QtCore.Qt.ControlModifier
                and event.modifiers() != QtCore.Qt.ShiftModifier
            ):
                if event.key() == QtCore.Qt.Key_Return:
                    if event.modifiers() == QtCore.Qt.AltModifier:
                        if len(self.imerge) > 1:
                            pass
                elif event.key() == QtCore.Qt.Key_Escape:
                    self.zoom_plot(1)
                    self.zoom_plot(3)
                    self.show()

                elif event.key() == QtCore.Qt.Key_Q:
                    self.viewbtns.button(0).setChecked(True)
                    self.viewbtns.button(0).press(self, 0)
                elif event.key() == QtCore.Qt.Key_W:
                    self.viewbtns.button(1).setChecked(True)
                    self.viewbtns.button(1).press(self, 1)
                elif event.key() == QtCore.Qt.Key_E:
                    self.viewbtns.button(2).setChecked(True)
                    self.viewbtns.button(2).press(self, 2)
                elif event.key() == QtCore.Qt.Key_R:
                    self.viewbtns.button(3).setChecked(True)
                    self.viewbtns.button(3).press(self, 3)
                elif event.key() == QtCore.Qt.Key_T:
                    self.viewbtns.button(4).setChecked(True)
                    self.viewbtns.button(4).press(self, 4)
                elif event.key() == QtCore.Qt.Key_U:
                    if "meanImg_chan2" in self.ops:
                        self.viewbtns.button(6).setChecked(True)
                        self.viewbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_Y:
                    if "meanImg_chan2_corrected" in self.ops:
                        self.viewbtns.button(5).setChecked(True)
                        self.viewbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_Space:
                    self.checkBox.toggle()
                # Agus
                elif event.key() == QtCore.Qt.Key_N:
                    self.checkBoxd.toggle()
                elif event.key() == QtCore.Qt.Key_B:
                    self.checkBoxn.toggle()
                elif event.key() == QtCore.Qt.Key_V:
                    self.checkBoxt.toggle()
                #
                elif event.key() == QtCore.Qt.Key_A:
                    self.colorbtns.button(0).setChecked(True)
                    self.colorbtns.button(0).press(self, 0)
                elif event.key() == QtCore.Qt.Key_S:
                    self.colorbtns.button(1).setChecked(True)
                    self.colorbtns.button(1).press(self, 1)
                elif event.key() == QtCore.Qt.Key_D:
                    self.colorbtns.button(2).setChecked(True)
                    self.colorbtns.button(2).press(self, 2)
                elif event.key() == QtCore.Qt.Key_F:
                    self.colorbtns.button(3).setChecked(True)
                    self.colorbtns.button(3).press(self, 3)
                elif event.key() == QtCore.Qt.Key_G:
                    self.colorbtns.button(4).setChecked(True)
                    self.colorbtns.button(4).press(self, 4)
                elif event.key() == QtCore.Qt.Key_H:
                    if self.hasred:
                        self.colorbtns.button(5).setChecked(True)
                        self.colorbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_J:
                    self.colorbtns.button(6).setChecked(True)
                    self.colorbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_K:
                    self.colorbtns.button(7).setChecked(True)
                    self.colorbtns.button(7).press(self, 7)
                elif event.key() == QtCore.Qt.Key_L:
                    if self.bloaded:
                        self.colorbtns.button(8).setChecked(True)
                        self.colorbtns.button(8).press(self, 8)
                elif event.key() == QtCore.Qt.Key_M:
                    if self.rastermap:
                        self.colorbtns.button(9).setChecked(True)
                        self.colorbtns.button(9).press(self, 9)
                elif event.key() == QtCore.Qt.Key_Left:
                    ctype = self.iscell[self.ichosen]
                    while -1:
                        self.ichosen = (self.ichosen - 1) % len(self.stat)
                        if self.iscell[self.ichosen] is ctype:
                            break
                    self.imerge = [self.ichosen]
                    self.ROI_remove()
                    self.update_plot()

    def update_plot(self):
        traces.plot_multiple_traces(self)
        self.show()


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
