import os, time
import numpy as np

from qtpy.QtWidgets import QFileDialog, QMessageBox

from .. import io

def load_dialog(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open stat.npy",
        "filter": "stat.npy",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    parent.fname = name[0]
    load_proc(parent)

def load_dialog_folder(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open folder with planeX folders",
    }    
    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    parent.fname = name
    load_folder(parent)


def load_folder(parent):
    print(parent.fname)
    save_folder = parent.fname
    plane_folders = [
        f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
    ]
    stat_found = False
    if len(plane_folders) > 0:
        stat_found = all(
            [os.path.isfile(os.path.join(f, "stat.npy")) for f in plane_folders])
    if not stat_found:
        print("No processed planeX folders in folder")
        return

    # create a combined folder to hold iscell and redcell
    output = io.combined(save_folder, save=False)
    parent.basename = os.path.join(parent.fname, "combined")
    load_to_GUI(parent, parent.basename, output)
    parent.loaded = True
    print(parent.fname)


def load_files(name):
    """ give stat.npy path and load all needed files for suite2p """
    try:
        stat = np.load(name, allow_pickle=True)
        ypix = stat[0]["ypix"]
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a stat.npy file :( "
              "(needs stat[n]['ypix']!)")
        stat = None
    goodfolder = False
    if stat is not None:
        basename, fname = os.path.split(name)
        goodfolder = True
        try:
            Fcell = np.load(basename + "/F.npy")
            Fneu = np.load(basename + "/Fneu.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: there are no fluorescence traces in this folder "
                  "(F.npy/Fneu.npy)")
            goodfolder = False
        try:
            Spks = np.load(basename + "/spks.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("there are no spike deconvolved traces in this folder "
                  "(spks.npy)")
            goodfolder = False
        try:
            ops = np.load(basename + "/ops.npy", allow_pickle=True).item()
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: there is no ops file in this folder (ops.npy)")
            goodfolder = False
        try:
            iscell = np.load(basename + "/iscell.npy")
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype("bool")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no manual labels found (iscell.npy)")
            if goodfolder:
                NN = Fcell.shape[0]
                iscell = np.ones((NN,), "bool")
                probcell = np.ones((NN,), np.float32)
        try:
            redcell = np.load(basename + "/redcell.npy")
            probredcell = redcell[:, 1].copy()
            redcell = redcell[:, 0].astype("bool")
            hasred = True
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no channel 2 labels found (redcell.npy)")
            hasred = False
            if goodfolder:
                NN = Fcell.shape[0]
                redcell = np.zeros((NN,), "bool")
                probredcell = np.zeros((NN,), np.float32)
    else:
        print("incorrect file, not a stat.npy")
        return None

    if goodfolder:
        return stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred
    else:
        print("stat.npy found, but other files not in folder")
        return None


def load_proc(parent):
    name = parent.fname
    print(name)
    basename, fname = os.path.split(name)
    output = load_files(name)
    if output is not None:
        load_to_GUI(parent, basename, output)
        parent.loaded = True
    else:
        Text = "Incorrect files, choose another?"
        load_again(parent, Text)


def load_to_GUI(parent, basename, procs):
    stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred = procs
    parent.basename = basename
    parent.stat = stat
    parent.ops = ops
    parent.Fcell = Fcell
    parent.Fneu = Fneu
    parent.Spks = Spks
    parent.iscell = iscell.astype("bool")
    parent.probcell = probcell
    parent.redcell = redcell.astype("bool")
    parent.probredcell = probredcell
    parent.hasred = hasred
    parent.notmerged = np.ones_like(parent.iscell).astype("bool")
    for n in range(len(parent.stat)):
        if parent.hasred:
            parent.stat[n]["chan2_prob"] = parent.probredcell[n]
        parent.stat[n]["inmerge"] = 0
    parent.stat = np.array(parent.stat)
    parent.ichosen = 0
    parent.imerge = [0]
    for n in range(len(parent.stat)):
        if "imerge" not in parent.stat[n]:
            parent.stat[n]["imerge"] = []


def save_merge(parent):
    print("saving to NPY")
    np.save(os.path.join(parent.basename, "ops.npy"), parent.ops)
    np.save(os.path.join(parent.basename, "stat.npy"), parent.stat)
    np.save(os.path.join(parent.basename, "F.npy"), parent.Fcell)
    np.save(os.path.join(parent.basename, "Fneu.npy"), parent.Fneu)
    if parent.hasred:
        np.save(os.path.join(parent.basename, "F_chan2.npy"), parent.F_chan2)
        np.save(os.path.join(parent.basename, "Fneu_chan2.npy"), parent.Fneu_chan2)
        np.save(
            os.path.join(parent.basename, "redcell.npy"),
            np.concatenate((np.expand_dims(
                parent.redcell, axis=1), np.expand_dims(parent.probredcell, axis=1)),
                           axis=1))
    np.save(os.path.join(parent.basename, "spks.npy"), parent.Spks)
    iscell = np.concatenate(
        (parent.iscell[:, np.newaxis], parent.probcell[:, np.newaxis]), axis=1)
    np.save(os.path.join(parent.basename, "iscell.npy"), iscell)

    parent.notmerged = np.ones(parent.iscell.size, "bool")



def load_again(parent, Text):
    tryagain = QMessageBox.question(parent, "ERROR", Text,
                                    QMessageBox.Yes | QMessageBox.No)

    if tryagain == QMessageBox.Yes:
        load_dialog(parent)