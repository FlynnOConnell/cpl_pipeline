import os, time

from qtpy.QtWidgets import QFileDialog, QMessageBox
from spk2extract import SpikeData, user_data_dir


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


def load_smr_to_GUI(parent, basename, output):
    parent.basename = basename
    parent.data = output
    parent.loaded = True
    parent.make_graphics()

def load_again(parent, Text):
    tryagain = QMessageBox.question(parent, "ERROR", Text,
                                    QMessageBox.Yes | QMessageBox.No)

    if tryagain == QMessageBox.Yes:
        load_smr(parent)