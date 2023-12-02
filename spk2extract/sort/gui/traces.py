import numpy as np
from pathlib import Path
from PyQt5.QtCore import QPoint, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QDialog,
    QComboBox,
    QPushButton,
)
from vispy import scene
from vispy.scene import AxisWidget
from vispy.scene.cameras import PanZoomCamera


class DataLoader(QThread):
    dataLoaded = pyqtSignal(object)

    def __init__(self, base_path):
        super(DataLoader, self).__init__()
        self.base_path = Path(base_path)
        self.data = {}
        if self.base_path.is_dir():
            print("Starting data thread...")
            self.start()

    def run(self):
        print("Loading data...")
        for file in self.base_path.iterdir():
            if file.is_dir():
                file_data = {}
                for channel_dir in (file / "Data").iterdir():
                    if channel_dir.is_dir():
                        channel_data = {}
                        for cluster_num in channel_dir.iterdir():
                            if cluster_num.is_dir():
                                cluster_data = {}
                                for cluster in cluster_num.iterdir():
                                    cluster_data[cluster.name] = {"path": str(cluster)}
                                channel_data[cluster_num.name] = cluster_data
                        for np_file in channel_dir.glob("*.npy"):
                            channel_data[np_file.stem] = {"path": str(np_file)}
                        file_data[channel_dir.name] = channel_data
                self.data[file.name] = file_data
        self.dataLoaded.emit(self.data)
        print("Data loaded!")


class ClusterSelectors(QComboBox):
    data_changed = pyqtSignal()

    def __init__(self, populate_func=None, **kwargs):
        super(ClusterSelectors, self).__init__(**kwargs)
        self.populate_func = populate_func
        self.currentIndexChanged.connect(self.emit_data_changed)

    def populate(self):
        self.clear()
        if self.populate_func:
            items = self.populate_func()
            if items:
                self.setEnabled(True)
                self.addItems(items)
            else:
                self.setEnabled(False)

    def emit_data_changed(self):
        self.data_changed.emit()


class YAxisPanZoomCamera(PanZoomCamera):
    """Custom camera that pans and zooms in the y-axis only for waveform visualizations."""

    def __init__(self, *args, **kwargs):
        super(YAxisPanZoomCamera, self).__init__(*args, **kwargs)

    def _update_pan(self, event):
        """Override to disable panning for x-axis."""
        dx = dy = 0
        p1 = event.last_event.pos
        p2 = event.pos
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dx /= self._viewbox.size[0]
        dy /= self._viewbox.size[1]
        self.pan((-dx, dy, 0, 0))

    def _update_zoom(self, event):
        """Override to disable zoom for x-axis."""
        s = 1.0 / 1.1
        if event.delta[1] > 0:
            s = 1.1

        # Modify s to make zoom only affect y-axis
        s = (1.0, s)

        ctr = np.array(self._viewbox.camera.rect.center)
        self.zoom((s[0], s[1]), center=ctr)


class PlotDialog(QDialog):
    def __init__(self, plot_widgets, parent=None):
        super(PlotDialog, self).__init__(parent)

        layout = QVBoxLayout()
        control_area = QWidget()
        control_layout = QHBoxLayout()
        control_area.setLayout(control_layout)

        normalize_button = QPushButton("Normalize Axis")
        normalize_button.clicked.connect(self.normalize_axis)
        control_layout.addWidget(normalize_button)

        plot_area = QWidget()
        main_plot_layout = QHBoxLayout()  # The main layout holding the columns
        plot_area.setLayout(main_plot_layout)

        column1_layout = QVBoxLayout()
        column2_layout = QVBoxLayout()

        toggle_column = True

        for key, widget in plot_widgets.items():
            title_label = QLabel(key)
            widget.setFixedHeight(300)
            if toggle_column:
                column1_layout.addWidget(title_label)
                column1_layout.addWidget(widget)
            else:
                column2_layout.addWidget(title_label)
                column2_layout.addWidget(widget)
            toggle_column = not toggle_column

        main_plot_layout.addLayout(column1_layout)
        main_plot_layout.addLayout(column2_layout)

        layout.addWidget(control_area)
        layout.addWidget(plot_area)
        self.setLayout(layout)

    def normalize_axis(self):
        """Normalize the y-axis of all plots to the same range."""
        mins, maxs = [], []
        for widget in self.findChildren(WaveformPlot):
            mins.append(widget.y_min)
            maxs.append(widget.y_max)
        min_val = min(mins)
        max_val = max(maxs)
        for widget in self.findChildren(WaveformPlot):
            widget.y_range = (min_val, max_val)



class WaveformPlot(QWidget):
    def __init__(self, parent, data, plot_title=None):
        super().__init__()
        self.parent = parent
        if data is None:
            self.data = self.parent.npy
        else:
            self.data = data
        self.plot_title = plot_title
        self.lines = None
        self.view = None
        self.grid = None
        self.y_min = float("inf")
        self.y_max = float("-inf")
        self.layout = QVBoxLayout(self)
        self.canvas = scene.SceneCanvas(show=True)
        self.layout.addWidget(self.canvas.native)
        self.setMinimumHeight(300)
        self.init_plot()

    @property
    def y_range(self):
        return self.view.camera.get_range()[1]

    @y_range.setter
    def y_range(self, y_range_tuple):
        self.view.camera.set_range(y=(y_range_tuple[0], y_range_tuple[1]))

    def update_line_color(self, new_color):
        self.lines.set_data(color=new_color)
        self.canvas.update()

    def init_plot(
        self,
    ):
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(
            aspect=None,
        )

        self.grid = self.canvas.central_widget.add_grid()
        self.grid.padding = 10
        self.grid.margin = 0

        x_axis = AxisWidget(
            orientation="bottom",
        )
        y_axis = AxisWidget(orientation="left")

        x_axis.stretch = (1, 0.1)
        y_axis.stretch = (0.01, 1)

        self.grid.add_widget(x_axis, row=1, col=0)
        self.grid.add_widget(y_axis, row=0, col=1)
        self.grid.add_widget(self.view, row=0, col=0)

        x_axis.link_view(self.view)
        y_axis.link_view(self.view)

        start_idx = 0
        end_idx = -1
        sub_npy = self.data[start_idx:end_idx, :]

        num_lines, num_samples = sub_npy.shape
        line_data = np.zeros((num_lines * num_samples, 2))

        # Create connections, reducing by the number of lines to disconnect them
        connect = np.ones(num_lines * num_samples, dtype=bool)

        # Set disconnection between separate lines
        for i in range(1, num_lines):
            individual_line_data = sub_npy[i, :]
            current_min = np.min(individual_line_data)
            current_max = np.max(individual_line_data)

            self.y_min = min(self.y_min, current_min)
            self.y_max = max(self.y_max, current_max)

            disconnect_idx = i * num_samples - 1  # The last point of each line
            connect[disconnect_idx] = False

        # Truncate & populate the "connect" array to match the line_data size
        connect = connect[:-num_lines]
        for i in range(num_lines):
            line_data[i * num_samples : (i + 1) * num_samples, 1] = sub_npy[i, :]
            line_data[i * num_samples : (i + 1) * num_samples, 0] = np.arange(
                num_samples
            )

        self.lines = scene.visuals.Line(
            pos=line_data, color="white", connect=connect, parent=self.view.scene
        )

        y_min = np.min(line_data[:, 1])
        y_max = np.max(line_data[:, 1])
        x_min = np.min(line_data[:, 0])
        x_max = np.max(line_data[:, 0])

        x_axis.axis.domain = (0, 100)
        x_axis.axis.pos = (0, x_max)
        x_axis.axis.axis_label = "Time (ms)"
        x_axis.axis.tick_color = "white"
        x_axis.axis.tick_label_margin = 10

        # no fmt
        self.view.camera.rect = (
            x_min,
            y_min,
            x_max,
            y_max,
        )
        self.view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))

        self.setLayout(self.layout)


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
