from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
from natsort import natsorted


def search_for_ext(rootdir, extension="tif", look_one_level_down=False):
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
        raise OSError("Could not find files, check path [{0}]".format(rootdir))


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


def get_h5_list(params: dict):
    """
    Make list of h5 files to process.

    if params["look_one_level_down"], then all h5"s in all folders + one level down

    Parameters
    ----------
    params : dict
        Parameters dictionary for the current recording.

    Returns
    -------
    fsall : list of str
        List of h5 files to process.
    params : dict
        Parameters dictionary for the current recording.

    """
    froot = params["data_path"]
    fold_list = params["data_path"]
    fsall = []
    nfs = 0
    first_tiffs = []
    for k, fld in enumerate(fold_list):
        fs, ftiffs = list_files(fld, params["look_one_level_down"],
                                ["*.h5", "*.hdf5", "*.mesc"])
        fsall.extend(fs)
        first_tiffs.extend(list(ftiffs))
    if len(fs) == 0:
        print("Could not find any h5 files")
        raise Exception("no h5s")
    else:
        params["first_tiffs"] = np.array(first_tiffs).astype("bool")
        print("** Found %d h5 files - converting to binary **" % (len(fsall)))
        #print("Found %d tifs"%(len(fsall)))
    return fsall, params


def get_nd2_list(params):
    """
    Make list of nd2 files to process
    if params["look_one_level_down"], then all nd2"s in all folders + one level down

    Parameters
    ----------
    params : dict
        Parameters dictionary for the current recording.

    Returns
    -------
    fsall : list of str
        List of nd2 files to process.

    """
    froot = params["data_path"]
    fold_list = params["data_path"]
    fsall = []
    nfs = 0
    first_tiffs = []
    for k, fld in enumerate(fold_list):
        fs, ftiffs = list_files(fld, params["look_one_level_down"], ["*.nd2"])
        fsall.extend(fs)
        first_tiffs.extend(list(ftiffs))
    if len(fs) == 0:
        print("Could not find any nd2 files")
        raise Exception("no nd2s")
    else:
        params["first_tiffs"] = np.array(first_tiffs).astype("bool")
        print("** Found %d nd2 files - converting to binary **" % (len(fsall)))
    return fsall, params


def find_files_open_binaries(params, ish5=False):
    """
    Finds tiffs or h5 files and opens binaries for writing

    Parameters
    ----------
    ish5 : bool
        Whether the input files are h5 files.
    params : list of dictionaries
        "keep_movie_raw", "data_path", "look_one_level_down", "reg_file"...

    Returns
    -------
    params1 : list of dictionaries
        adds fields "filelist", "first_tiffs", opens binaries

    """

    reg_file = []
    reg_file_chan2 = []

    for param in params:
        nchannels = param["nchannels"]
        if "keep_movie_raw" in param and param["keep_movie_raw"]:
            reg_file.append(open(param["raw_file"], "wb"))
            if nchannels > 1:
                reg_file_chan2.append(open(param["raw_file_chan2"], "wb"))
        else:
            reg_file.append(open(param["reg_file"], "wb"))
            if nchannels > 1:
                reg_file_chan2.append(open(param["reg_file_chan2"], "wb"))

        if "input_format" in param.keys():
            input_format = param["input_format"]
        else:
            input_format = "tif"
    if ish5:
        input_format = "h5"
    print(input_format)
    if input_format == "h5":
        if len(params[0]["data_path"]) > 0:
            fs, params2 = get_h5_list(params[0])
            print("NOTE: using a list of h5 files:")
            print(fs)
        # find h5"s
        else:
            if params[0]["look_one_level_down"]:
                fs = list_h5(params[0])
                print("NOTE: using a list of h5 files:")
                print(fs)
            else:
                fs = [params[0]["h5py"]]
    elif input_format == "sbx":
        # find sbx
        fs, params2 = get_sbx_list(params[0])
        print("Scanbox files:")
        print("\n".join(fs))
    elif input_format == "nd2":
        # find nd2s
        fs, params2 = get_nd2_list(params[0])
        print("Nikon files:")
        print("\n".join(fs))
    else:
        raise Exception("input_format not recognized")
    for param in params:
        param["filelist"] = fs
    return params, fs, reg_file, reg_file_chan2


def init_params(params):
    """ initializes params files for each plane in recording

    Parameters
    ----------
    params : dictionary
        "nplanes", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw"
        + (if mesoscope) "dy", "dx", "lines"

    Returns
    -------
        params1 : list of dictionaries
            adds fields "save_path0", "reg_file"
            (depending on params: "raw_file", "reg_file_chan2", "raw_file_chan2")

    """

    dx, dy = None, None
    lines = []
    nplanes = params["nplanes"]
    nchannels = params["nchannels"]
    if "lines" in params:
        lines = params["lines"]
    if "iplane" in params:
        iplane = params["iplane"]
        #params["nplanes"] = len(params["lines"])
    params1 = []
    if ("fast_disk" not in params) or len(params["fast_disk"]) == 0:
        params["fast_disk"] = params["save_path0"]
    fast_disk = params["fast_disk"]
    # for mesoscope recording FOV locations
    if "dy" in params and params["dy"] != "":
        dy = params["dy"]
        dx = params["dx"]
    # compile params into list across planes
    for j in range(0, nplanes):
        if len(params["save_folder"]) > 0:
            params["save_path"] = os.path.join(params["save_path0"], params["save_folder"],
                                            "plane%d" % j)
        else:
            params["save_path"] = os.path.join(params["save_path0"], "spk2extract", "plane%d" % j)

        if ("fast_disk" not in params) or len(params["fast_disk"]) == 0:
            params["fast_disk"] = params["save_path0"].copy()
        fast_disk = os.path.join(params["fast_disk"], "spk2extract", "plane%d" % j)
        params["params_path"] = os.path.join(params["save_path"], "params.npy")
        params["reg_file"] = os.path.join(fast_disk, "data.bin")
        if "keep_movie_raw" in params and params["keep_movie_raw"]:
            params["raw_file"] = os.path.join(fast_disk, "data_raw.bin")
        if "lines" in params:
            params["lines"] = lines[j]
        if "iplane" in params:
            params["iplane"] = iplane[j]
        if nchannels > 1:
            params["reg_file_chan2"] = os.path.join(fast_disk, "data_chan2.bin")
            if "keep_movie_raw" in params and params["keep_movie_raw"]:
                params["raw_file_chan2"] = os.path.join(fast_disk, "data_chan2_raw.bin")
        if "dy" in params and params["dy"] != "":
            params["dy"] = dy[j]
            params["dx"] = dx[j]
        if not os.path.isdir(fast_disk):
            os.makedirs(fast_disk)
        if not os.path.isdir(params["save_path"]):
            os.makedirs(params["save_path"])
        params1.append(params.copy())
    return params1


def get_spk2extract_path(path: Path | str) -> Path:
    """
    Get the path to the root spk2extract folder from a path to a file or folder in the spk2extract folder.
    """

    new_path = None
    path = Path(path)  # In case `path` is a string

    # Cheap sanity check
    if "spk2extract" in str(path):
        # Walk the folders in path backwards
        for path_idx in range(len(path.parts) - 1, 0, -1):
            if path.parts[path_idx] == "spk2extract":
                new_path = Path(path.parts[0])
                for path_part in path.parts[1:path_idx + 1]:
                    new_path = new_path.joinpath(path_part)
                break
    else:
        raise FileNotFoundError("The `spk2extract` folder was not found in path")
    return new_path
