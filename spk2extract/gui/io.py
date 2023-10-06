import os, time
import numpy as np

from PyQt5.QtWidgets import QFileDialog, QMessageBox
from spk2extract import SpikeData

def select_base_folder(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Select base folder",
        "options": QFileDialog.DontUseNativeDialog,
    }
    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    parent.base_path = name

def load_smr(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open .smr file",
        "filter": ".smr",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    basename, fname = os.path.split(str(name))
    parent.fname = name[0]
    spike_data = SpikeData(parent.fname)
    spike_data.extract()
    load_smr_to_GUI(parent, basename, spike_data)

def load_npy(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open .npy file",
        "filter": ".npy",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    basename, fname = os.path.split(str(name))
    parent.fname = name[0]
    data = np.load(parent.fname)
    load_npy_to_GUI(parent, basename, data)

def load_smr_to_GUI(parent, basename, output):
    parent.basename = basename
    parent.data = output
    parent.loaded = True
    parent.make_graphics()

def load_npy_to_GUI(parent, basename, output):
    parent.basename = basename
    parent.npy = output
    parent.loaded = True
    parent.make_graphics_npy()

def load_again(parent, Text):
    tryagain = QMessageBox.question(parent, "ERROR", Text,
                                    QMessageBox.Yes | QMessageBox.No)

    if tryagain == QMessageBox.Yes:
        load_smr(parent)