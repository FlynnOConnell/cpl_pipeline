import numpy as np
import os, re

support_rec_types = {
    "one file per channel": "amp-\S-\d*\.dat",
    "one file per signal type": "amplifier\.dat",
}
voltage_scaling = 0.195


def read_digital_dat(file_dir, dig_channels=None, dig_type="in"):
    """Reads digitalin.dat from intan recording with file_type 'one file per signal type'

    Parameters
    ----------
    file_dir : str, file directory for recording data
    dig_channels : list (optional), digital channel numbers to get
    dig_type : {'in','out'}, type of digital signal to get (default 'in')

    Returns
    -------
    numpy.ndarray : one row per digital_input channel corresponding to dig_in
                    from rec_info

    Throws
    ------
    FileNotFoundError : if digitalin.dat is not in file_dir
    """
    if dig_channels is None:
        # rec_info = read_rec_info(file_dir)
        # dig_channels = rec_info['dig_%s' % dig_type]
        pass
    dat_file = os.path.join(file_dir, "digital%s.dat" % dig_type)
    file_dat = np.fromfile(dat_file, dtype=np.dtype("int16"))
    chan_dat = []
    for ch in dig_channels:
        tmp_dat = (file_dat & pow(2, ch) > 0).astype(np.dtype("uint16"))
        chan_dat.append(tmp_dat)
    out = np.array(chan_dat)
    return out


def read_one_channel_file(file_name):
    """Reads a single amp or din channel file created by an intan 'one file per
    channel' recording

    Parameters
    ----------
    file_name : str, absolute path to file to read data from

    Returns
    -------
    numpy.ndarray : int16, 1D array of data from amp file

    Throws
    ------
    FileNotFoundError : if file_name is not found
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError("Could not locate file %s" % file_name)

    chan_dat = np.fromfile(file_name, dtype=np.dtype("int16"))
    return chan_dat


def get_recording_filetype(file_dir):
    """Check Intan recording directory to determine type of recording and thus
    extraction method to use. Asks user to confirm, and manually correct if
    incorrect

    Parameters
    ----------
    file_dir : str, recording directory to check

    Returns
    -------
    str : file_type of recording
    """
    file_list = os.listdir(file_dir)
    file_type = None
    for k, v in support_rec_types.items():
        regex = re.compile(v)
        if any([True for x in file_list if regex.match(x) is not None]):
            file_type = k

    if file_type is None:
        msg = "\n   ".join(
            [
                "unsupported recording type. Supported types are:",
                *list(support_rec_types.keys()),
            ]
        )
    else:
        msg = '"' + file_type + '"'

    return file_type

    # Removing query since this is pretty accurate
    # query = 'Detected recording type is %s \nIs this correct?:  ' % msg
    # q = userIO.ask_user(query,,
    #                    shell=shell)

    # if q == 1:
    #    return file_type
    # else:
    #    choice = userIO.select_from_list('Select correct recording type',
    #                                     list(support_rec_types.keys()),
    #                                     'Select Recording Type',
    #                                     shell=shell)
    #    choice = list(support_rec_types.keys())[choice]
    #    return choice
