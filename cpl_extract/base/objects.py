from __future__ import annotations

import os
from pathlib import Path

from cpl_extract import logger
from cpl_extract.spk_io import prompt
import pickle


class data_object:
    def __init__(
            self,
            data_type: str = None,
            root_dir: str | Path = None,
            data_name: str = None,
            save_file: str | Path = None,
            log_file: str | Path = None,
            shell: bool = False,
    ):
        if "SSH_CONNECTION" in os.environ:
            shell = True

        # use the class name as the data_type if not provided
        if data_type is None:
            logger.cpl_logger.info("No data_type provided, using class name.")
            data_type = self.__class__.__name__.lower()

        if root_dir is None:
            root_dir = prompt.get_filedirs("Select %s directory" % data_type, shell=shell)
            if root_dir is None or not os.path.isdir(root_dir):
                raise NotADirectoryError("Must provide a valid root directory for the %s" % data_type)

        if data_name is None:
            data_name = prompt.get_user_input("Enter name for %s" % data_type, os.path.basename(root_dir), shell)

        if save_file is None:
            save_file = os.path.join(root_dir, "%s_%s.p" % (data_name, data_type))

        if log_file is None:
            # check globals for logfile
            if f"{data_name}_{data_type}.log" in globals():
                log_file = globals()["cpl_extract_logfile"]
                logger.cpl_logger.info(f"Using global logfile {log_file}.")
            elif f"{data_name}.log" in globals():
                log_file = globals()["cpl_extract_logfile"]
                logger.cpl_logger.info(f"Using global logfile {log_file}.")
            else:
                log_file = Path().home() / "cpl_extract" / "logs" / f"{data_name}_{data_type}.log"
                logger.cpl_logger.info(f"Using default logfile {log_file}.")
                globals()[str(log_file)] = log_file
            if not os.path.isfile(log_file):
                # create log file
                log_file.parent.mkdir(parents=True, exist_ok=True)
                open(log_file, "w").close()
                logger.cpl_logger.info(f"Created logfile {log_file}.")
        self.root_dir = root_dir
        self.data_type = data_type
        self.data_name = data_name
        self.save_file = save_file
        self.log_file = log_file

    def save(self):
        """Saves the data_object to a .p file"""
        if not self.save_file.endswith(".p"):
            self.save_file += ".p"
        if not Path(self.save_file).parent.exists():
            print(f"Creating directory {self.save_file.parent}...for saving the pickled data object.")
            Path(self.save_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saving {self.data_type}: {self.data_name}... \n"
                  f"Saving to {self.save_file}")

    def _change_root(self, new_root=None):
        if "SSH_CONNECTION" in os.environ:
            shell = True
        else:
            shell = False

        if new_root is None:
            new_root = prompt.get_filedirs(
                "Select new location of %s" % self.root_dir, shell=shell
            )

        old_root = self.root_dir
        self.root_dir = self.root_dir.replace(old_root, new_root)
        self.save_file = self.save_file.replace(old_root, new_root)
        self.log_file = self.log_file.replace(old_root, new_root)
        return new_root

    def __str__(self):
        out = []
        out.append(self.data_type + " :: " + self.data_name)
        out.append("Root Directory : %s" % self.root_dir)
        out.append("Save File : %s" % self.save_file)
        out.append("Log File : %s" % self.log_file)
        return "\n".join(out)

    def export_to_txt(self):
        sf = self.save_file.replace(".p", ".txt")
        with open(sf, "w") as f:
            print(self, file=f)


def load_data(data_type, file_dir=None, shell=False):
    """Loads a data_object .p file and returns the object

    Parameters
    ----------
    data_type : str
        type of data_object extension do you want
        dataset, experiment or object
    file_dir : str (optional)
        path to file dir that the .p file is saved in
        or path to .p file

    Returns
    -------
    blechpy.data_object

    Raises
    ------
    NotADirectoryError
    """
    if "SSH_CONNECTION" in os.environ:
        shell = True

    if file_dir is None:
        file_dir = prompt.get_filedirs(
            "Select %s directory or .p file" % data_type, shell=shell
        )

    if os.path.isfile(file_dir) and f"{data_type}.p" in file_dir:
        data_file = [file_dir]
        file_dir = os.path.dirname(file_dir)
    elif not os.path.isdir(file_dir):
        raise NotADirectoryError("%s not found." % file_dir)
    else:
        data_file = [x for x in os.listdir(file_dir) if x.endswith("%s.p" % data_type)]

    if len(data_file) == 0:
        return None
    elif len(data_file) > 1:
        tmp = prompt.select_from_list(
            "Multiple %s files found." "Select the one you want to load." % data_type,
            data_file,
            shell=shell,
        )
        if tmp is None:
            return None
        else:
            data_file = tmp
    else:
        data_file = data_file[0]

    data_file = os.path.join(file_dir, data_file)
    with open(data_file, "rb") as f:
        out = pickle.load(f)

    return out


def load_experiment(file_dir=None, shell=False):
    """Loads experiment.p file from file_dir

    Parameters
    ----------
    file_dir : str (optional), if not provided, file chooser will appear

    Returns
    -------
    blechpy.experiment or None if no file found
    """
    return load_data("experiment", file_dir, shell=shell)


def load_dataset(file_dir=None, shell=False):
    """Loads dataset.p file from file_dir

    Parameters
    ----------
    file_dir : str (optional), if not provided, file chooser will appear

    Returns
    -------
    blechpy.dataset or None if no file found
    """
    return load_data("dataset", file_dir, shell=shell)


def load_project(file_dir=None, shell=False):
    """Loads project.p file from file_dir

    Parameters
    ----------
    file_dir : str (optional), if not provided, file chooser will appear

    Returns
    -------
    blechpy.project or None if no file found
    """
    return load_data("project", file_dir, shell=shell)


def load_pickled_object(fn):
    with open(fn, "rb") as f:
        out = pickle.load(f)

    return out
