import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QPoint, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtWidgets import QWidget


class DataPreparationThread(QThread):
    data_ready = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, npy, start_idx, end_idx):
        super(DataPreparationThread, self).__init__()
        self.npy = npy
        self.start_idx = start_idx
        self.end_idx = end_idx

    def run(self):
        sub_npy = self.npy[self.start_idx:self.end_idx, :]
        x = np.arange(sub_npy.shape[1])
        self.data_ready.emit(x, sub_npy)


class QRangeSlider(QWidget):
    range_changed = pyqtSignal(int, int)

    def __init__(self, min_val=0, max_val=100, *args, **kwargs):
        super(QRangeSlider, self).__init__(*args, **kwargs)

        self.min_val = min_val
        self.max_val = max_val
        self.left_value = self.min_val
        self.right_value = self.max_val

        self.pressed_control = None
        self.hover_control = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressed_control = self.get_control(event.pos())
            self.click_offset = self.get_pos(self.pressed_control) - event.pos().x()

    def mouseMoveEvent(self, event):
        if self.pressed_control:
            value = self.pixel_to_value(
                event.pos().x() + self.click_offset, self.pressed_control
            )
            self.set_value(value, self.pressed_control)

    def mouseReleaseEvent(self, event):
        self.pressed_control = None

    def get_control(self, pos):
        left_pos = self.get_pos("left")
        right_pos = self.get_pos("right")

        if abs(left_pos - pos.x()) < abs(right_pos - pos.x()):
            return "left"
        else:
            return "right"

    def get_pos(self, control):
        if control == "left":
            return self.value_to_pixel(self.left_value)
        return self.value_to_pixel(self.right_value)

    def value_to_pixel(self, val):
        return int(self.width() * (val - self.min_val) / (self.max_val - self.min_val))

    def pixel_to_value(self, pos, control):
        return self.min_val + (pos / self.width()) * (self.max_val - self.min_val)

    def set_value(self, val, control):
        if control == "left":
            self.left_value = max(min(val, self.right_value), self.min_val)
        else:
            self.right_value = min(max(val, self.left_value), self.max_val)
        self.range_changed.emit(self.left_value, self.right_value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Draw the background line
        painter.setPen(QPen(Qt.gray, 1))
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        for control, color in [("left", Qt.red), ("right", Qt.green)]:
            painter.setPen(QPen(color, 3))
            painter.drawEllipse(QPoint(self.get_pos(control), self.height() // 2), 5, 5)

class MultiLine(pg.GraphicsObject):
    def __init__(self, x, y, *args, **kwargs):
        pg.GraphicsObject.__init__(self, *args)
        self.x = x
        self.y = y
        self.generate_picture()

    def generate_picture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen("w"))

        for i in range(self.y.shape[0]):
            path = pg.arrayToQPath(self.x, self.y[i])
            p.drawPath(path)

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

def plot_npy_trace(parent):
    p = parent.plotWidgets["npy"]
    p.clear()

    num_waveforms, num_points = parent.npy.shape
    x = np.arange(num_points)

    multi_line = MultiLine(x, parent.npy)
    p.addItem(multi_line)
    p.autoRange()

# Update plot_multiple_traces
def plot_multiple_traces(parent):
    for key in parent.data.data.keys():
        p = parent.plotWidgets[key]
        p.clear()

        unit_data = parent.data[key]
        spikes = np.array(unit_data.spikes)  # assuming spikes is a 2D array
        times = np.array(unit_data.times)  # assuming times is a 1D array

        times = np.repeat(times[:, np.newaxis], spikes.shape[1], axis=1)

        # Make a single 1D array for both spikes and times
        spikes = spikes.flatten()
        times = times.flatten()

        p.plot(times, spikes, pen="w")
