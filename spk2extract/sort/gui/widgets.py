from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QDialog, QListWidget, QAbstractItemView,
                             QVBoxLayout, QPushButton)


class MultiSelectionDialog(QDialog):
    selection_made = pyqtSignal(list)

    def __init__(self, populate_func=None, parent=None):
        super(MultiSelectionDialog, self).__init__(parent)
        self.populate_func = populate_func
        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.close_button = QPushButton("Select/Close")
        self.close_button.clicked.connect(self.close_and_emit)

        layout.addWidget(self.list_widget)
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.populate()

    def populate(self):
        self.list_widget.clear()
        if self.populate_func:
            items = self.populate_func()
            self.list_widget.addItems(items)

    def close_and_emit(self):
        selected_items = [item.text() for item in self.list_widget.selectedItems()]
        self.selection_made.emit(selected_items)
        self.close()
