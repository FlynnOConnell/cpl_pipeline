import numpy as np
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QLabel, QComboBox, QPushButton, QLineEdit, QCheckBox
from pyqtgraph import PlotItem


# Update plot_multiple_traces
def plot_multiple_traces(parent):
    for key in parent.data.data.keys():
        p = parent.plotWidgets[key]  # get corresponding PlotWidget
        p.clear()

        unit_data = parent.data[key]
        spikes = np.array(unit_data.spikes)  # assuming spikes is a 2D array
        times = np.array(unit_data.times)  # assuming times is a 1D array

        # Repeat the times to match the spikes, assuming spikes has shape (n_spikes, n_points)
        times = np.repeat(times[:, np.newaxis], spikes.shape[1], axis=1)

        # Make a single 1D array for both spikes and times
        spikes = spikes.flatten()
        times = times.flatten()

        p.plot(times, spikes, pen="w")



def make_buttons(parent, b0):
    # combo box to decide what kind of activity to view
    qlabel = QLabel(parent)
    qlabel.setText("<font color='white'>Activity mode:</font>")
    parent.l0.addWidget(qlabel, b0, 0, 1, 1)
    parent.comboBox = QComboBox(parent)
    parent.comboBox.setFixedWidth(100)
    parent.l0.addWidget(parent.comboBox, b0 + 1, 0, 1, 1)
    parent.comboBox.addItem("F")
    parent.comboBox.addItem("Fneu")
    parent.comboBox.addItem("F - 0.7*Fneu")
    parent.comboBox.addItem("deconvolved")
    parent.activityMode = 3
    parent.comboBox.setCurrentIndex(parent.activityMode)

    # up/down arrows to resize view
    parent.level = 1
    parent.arrowButtons = [
        QPushButton(u" \u25b2"),
        QPushButton(u" \u25bc"),
    ]
    parent.arrowButtons[0].clicked.connect(lambda: expand_trace(parent))
    parent.arrowButtons[1].clicked.connect(lambda: collapse_trace(parent))
    b = 0
    for btn in parent.arrowButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(btn, b0 + b, 1, 1, 1, QtCore.Qt.AlignRight)
        b += 1

    parent.pmButtons = [QPushButton(" +"), QPushButton(" -")]
    parent.pmButtons[0].clicked.connect(lambda: expand_scale(parent))
    parent.pmButtons[1].clicked.connect(lambda: collapse_scale(parent))
    b = 0
    parent.sc = 2
    for btn in parent.pmButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(btn, b0 + b, 1, 1, 1)
        b += 1
    # choose max # of cells plotted
    parent.l0.addWidget(
        QLabel("<font color='white'>max # plotted:</font>"),
        b0 + 2,
        0,
        1,
        1,
    )
    b0 += 3
    parent.ncedit = QLineEdit(parent)
    parent.ncedit.setValidator(QtGui.QIntValidator(0, 400))
    parent.ncedit.setText("40")
    parent.ncedit.setFixedWidth(35)
    parent.ncedit.setAlignment(QtCore.Qt.AlignRight)
    parent.ncedit.returnPressed.connect(lambda: nc_chosen(parent))
    parent.l0.addWidget(parent.ncedit, b0, 0, 1, 1)
    # traces CHECKBOX
    parent.l0.setVerticalSpacing(4)
    parent.checkBoxt = QCheckBox("raw fluor [V]")
    parent.checkBoxt.setStyleSheet("color: cyan;")
    parent.checkBoxt.toggled.connect(lambda: traces_on(parent))
    parent.tracesOn = True
    parent.checkBoxt.toggle()
    parent.l0.addWidget(parent.checkBoxt, b0, 7, 1, 2)
    return b0


def expand_scale(parent):
    parent.sc += 0.5
    parent.sc = np.minimum(10, parent.sc)
    plot_multiple_traces(parent)
    parent.show()


def collapse_scale(parent):
    parent.sc -= 0.5
    parent.sc = np.maximum(0.5, parent.sc)
    plot_multiple_traces(parent)
    parent.show()


def expand_trace(parent):
    parent.level += 1
    parent.level = np.minimum(5, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)
    #parent.p1.zoom_plot()


def collapse_trace(parent):
    parent.level -= 1
    parent.level = np.maximum(1, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)
    #parent.p1.zoom_plot()


def nc_chosen(parent):
    if parent.loaded:
        plot_multiple_traces(parent)
        parent.show()

def traces_on(parent):
    state = parent.checkBoxt.isChecked()
    if parent.loaded:
        if state:
            parent.tracesOn = True
        else:
            parent.tracesOn = False
        plot_multiple_traces(parent)
        parent.win.show()
        parent.show()