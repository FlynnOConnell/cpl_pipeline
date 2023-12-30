from PyQt5.QtWidgets import QFileDialog, QMessageBox

from cpl_pipeline import load_pickled_object

def select_base_folder(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Select base folder",
        "options": QFileDialog.DontUseNativeDialog,
    }

    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    parent.base_path = name