from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from natsort import natsorted


def search_for_ext(rootdir, extension="h5", look_one_level_down=False):
    filepaths = []
    if os.path.isdir(rootdir):
        # search root dir
        tmp = glob.glob(os.path.join(rootdir, "*." + extension))
        if len(tmp):
            filepaths.extend([t for t in natsorted(tmp)])
        # search one level down
        if look_one_level_down:
            dirs = natsorted(os.listdir(rootdir))
            for d in dirs:
                if os.path.isdir(os.path.join(rootdir, d)):
                    tmp = glob.glob(os.path.join(rootdir, d, "*." + extension))
                    if len(tmp):
                        filepaths.extend([t for t in natsorted(tmp)])
    if len(filepaths):
        return filepaths
    else:
        raise OSError(f"Could not find files, check path [{rootdir}]")


def get_sbx_list(params):
    """ make list of scanbox files to process
    if params["subfolders"], then all tiffs params["data_path"][0] / params["subfolders"] / *.sbx
    if params["look_one_level_down"], then all tiffs in all folders + one level down
    TODO: Implement "tiff_list" functionality
    """
    froot = params["data_path"]
    # use a user-specified list of tiffs
    if len(froot) == 1:
        if "subfolders" in params and len(params["subfolders"]) > 0:
            fold_list = []
            for folder_down in params["subfolders"]:
                fold = os.path.join(froot[0], folder_down)
                fold_list.append(fold)
        else:
            fold_list = params["data_path"]
    else:
        fold_list = froot
    fsall = []
    for k, fld in enumerate(fold_list):
        fs = search_for_ext(fld, extension="sbx",
                            look_one_level_down=params["look_one_level_down"])
        fsall.extend(fs)
    if len(fsall) == 0:
        print(fold_list)
        raise Exception("No files, check path.")
    else:
        print("** Found %d sbx - converting to binary **" % (len(fsall)))
    return fsall, params


def list_h5(params):
    froot = os.path.dirname(params["h5py"])
    lpath = os.path.join(froot, "*.h5")
    fs = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.hdf5")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    return fs


def list_files(froot, look_one_level_down, exts):
    """
    Get list of files with exts in folder froot + one level down.

    Parameters
    ----------
    froot : str
        Path to folder to search.
    look_one_level_down : bool
        Whether to search one level down.
    exts : list of str
        List of file extensions to search for.
    """
    fs = []
    for e in exts:
        lpath = os.path.join(froot, e)
        fs.extend(glob.glob(lpath))
    fs = natsorted(set(fs))
    if len(fs) > 0:
        first_tiffs = np.zeros((len(fs),), "bool")
        first_tiffs[0] = True
    else:
        first_tiffs = np.zeros(0, "bool")
    lfs = len(fs)
    if look_one_level_down:
        fdir = natsorted(glob.glob(os.path.join(froot, "*/")))
        for folder_down in fdir:
            fsnew = []
            for e in exts:
                lpath = os.path.join(folder_down, e)
                fsnew.extend(glob.glob(lpath))
            fsnew = natsorted(set(fsnew))
            if len(fsnew) > 0:
                fs.extend(fsnew)
                first_tiffs = np.append(first_tiffs, np.zeros((len(fsnew),), "bool"))
                first_tiffs[lfs] = True
                lfs = len(fs)
    return fs, first_tiffs


def get_spk2extract_path(path: Path | str) -> Path:
    """
    Get the path to the root spk2extract folder from a path to a file or folder in the spk2extract folder.
    """

    new_path = None
    path = Path(path)
    if "spk2extract" in str(path):
        for path_idx in range(len(path.parts) - 1, 0, -1):
            if path.parts[path_idx] == "spk2extract":
                new_path = Path(path.parts[0])
                for path_part in path.parts[1:path_idx + 1]:
                    new_path = new_path.joinpath(path_part)
                break
    else:
        raise FileNotFoundError("The `spk2extract` folder was not found in path")
    return new_path
