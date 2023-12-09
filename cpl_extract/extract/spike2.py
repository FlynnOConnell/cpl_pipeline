"""
Spike data extract utility, the main workhorse of this package.
"""
from __future__ import annotations

import string
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sonpy import lib as sp
except ImportError:
    try:
        import sonpy as sp
    except:
        pass
        # raise Warning("sonpy not found. Are you on a M1 Mac?")

import tables
from cpl_extract import logger as cpl_logger
from cpl_extract import spk_io
from cpl_extract.utils import check_substring_content

# any type that contains "Mark" or "mark" is an event channel:
# - Marker
# - RealMark
# - TextMark
# - AdcMark ** not sure what this is, ReadMarkers() doesn't work on it

EVENTS = [
    sp.DataType.Marker,
    sp.DataType.RealMark,
    sp.DataType.TextMark,
    sp.DataType.RealMark,
]

ADC = [sp.DataType.Adc]


class Channel:
    def __init__(self, name, chan_type, chan_data, times):
        self.name = name
        self.type = chan_type
        self.data: np.ndarray = chan_data
        self.times: np.ndarray = times

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"


class SonfileException(BaseException):
    """
    Exception for sonfile being wrong filetype.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


def is_ascii_letter(char):
    if char in range(65, 91) or char in range(97, 123):
        return True


def codes_to_string(codes):
    return "".join(chr(code) for code in codes if code != 0)


def indices_to_time(indices, fs):
    """Spike2 indices are in clock ticks, this converts them to seconds."""
    return np.array(indices) / float(fs)


def ticks_to_time(ticks, time_base):
    """Converts clock ticks to seconds."""
    return np.array(ticks) * time_base


class Spike2Data:
    """
    Class for reading and storing data from a Spike2 file.

    Parameters:
    -----------
    filepath : Path | str
        The full path to the Spike2 file, including filename + extension.
    exclude : tuple, optional
        A tuple of channel names to exclude from the data. Default is empty tuple.

    Attributes:
    -----------
    exclude : tuple
        A tuple of channel names to exclude from the data.
    filename : Path
        The full path to the Spike2 file, including filename + extension.
    sonfile : SonFile
        The SonFile object from the sonpy library.
    unit : dict
        A dictionary of unit channels, where the keys are the channel names and the values are the waveforms.
    time_base : float
        Everything in the file is quantified by the underlying clock tick (64-bit).
        All values in the file are stored, set and returned in ticks.
        You need to read this value to interpret times in seconds.
    max_time : float
        The last time-point in the array, in ticks.
    max_channels : int
        The number of channels in the file.
    bitrate : int
        Whether the file is 32bit (old) or 64bit (new).
    recording_length : float
        The total recording length, in seconds.

    Notes
    -----
    This class can be used as if it were several different types of objects:
    - A dictionary, where the keys are the channel names and the values are the waveforms.
    - A list, where the elements are the channel names.
    - A string, where the string is the filename stem.

    Examples
    --------
    >>> from cpl_extract.extract import Spike2Data
    >>> from pathlib import Path
    >>> smr_path = Path().home() / "data" / "smr"
    >>> files = [f for f in smr_path.glob("*.smr")]
    >>> data = Spike2Data(files[0])
    >>> data
    'rat1-2021-03-24_0001'
    >>> data["LFP1"]
    WaveData(spikes=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00],), times=array([...]))

    >>> data["LFP1"].data
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,]])
    >>> data["LFP1"].times
    array([0.00000000e+00,  0.00000000e+00,  0.00000000e+00,])

    """

    @property
    def logger(self):
        return self._logger

    def __init__(
        self,
        filepath: Path | str,
        savepath: Path | str,
        extraction_logger: cpl_logger.logger = None,
        log_level="INFO",
    ):
        """
        Class for reading and storing data from a Spike2 file.

        Parameters:
        -----------
        filepath : Path | str
            The full path to the Spike2 file, including filename + extension.
        savepath : Path | str, optional
            The full path to the directory to save the data to. Default is None.
        extraction_logger : logger, optional
            The logger to use for extraction. Default is None.
        exclude : tuple, optional
            A tuple of channel names to exclude from the data. Default is empty tuple.

        Attributes:
        -----------
        exclude : tuple
            A tuple of channel names to exclude from the data.
        filename : Path
            The full path to the Spike2 file, including filename + extension.
        sonfile : SonFile
            The SonFile object from the sonpy library.
        unit : dict
            A dictionary of unit channels, where the keys are the channel names and the values are the waveforms.
        preinfusion : bool
            Whether the file is a pre-infusion file or not.
        postinfusion : bool
            Whether the file is a post-infusion file or not.
        time_base : float
            Everything in the file is quantified by the underlying clock tick (64-bit).
            All values in the file are stored, set and returned in ticks.
            You need to read this value to interpret times in seconds.
        max_time : float
            The last time-point in the array, in ticks.
        max_channels : int
            The number of channels in the file.
        bitrate : int
            Whether the file is 32bit (old) or 64bit (new).
        recording_length : float
            The total recording length, in seconds.
        """

        if extraction_logger is None:
            self._logger = cpl_logger.logger
        else:
            self._logger = extraction_logger
        self._log_level = log_level
        self.errors = {}
        self.filename = Path(filepath)
        self._savedir = Path(savepath) / self.filename.stem

        self.sonfile = sp.SonFile(str(self.filename), True)

        self.data = pd.DataFrame()
        self.data["idx"] = range(self._max_channels())  # the index stored by sonpy
        self.data["name"] = [
            self.sonfile.GetChannelTitle(idx) for idx in range(self._max_channels())
        ]
        self.data["type"] = [
            self.sonfile.ChannelType(idx) for idx in range(self._max_channels())
        ]
        self.data["fs"] = [
            np.round(1 / (self.sonfile.ChannelDivide(idx) * self._time_base()), 2)
            for idx in range(self._max_channels())
        ]
        self.data["units"] = [
            self.sonfile.GetChannelUnits(idx) for idx in range(self._max_channels())
        ]
        self.rec_length = self._max_time()
        self.bitrate = 32 if self.sonfile.is32file() else 64

        self.events = self.data[self.data["type"].isin(EVENTS)]
        self.waves = self.data[self.data["type"].isin(ADC)]
        # separate out waves by the name containing "u" or "L" (for unit or LFP) using where check_substring_content(self.waves["name"], "u")
        self.units = self.waves[
            self.waves["name"].apply(lambda x: check_substring_content(x, "u"))
        ]
        self.lfps = self.waves[
            self.waves["name"].apply(lambda x: check_substring_content(x, "lfp"))
        ]

        # filter out any types = DataType.Off
        self.data = self.data[self.data["type"] != sp.DataType.Off]
        self.TIME_EXTRACTED = False

        if self.sonfile.GetOpenError() != 0:
            if self.filename.suffix not in [".smr", ".smrx"]:
                raise SonfileException(
                    f"{self.filename} is not a valid file. \n"
                    f"Extension {self.filename.suffix} is not valid."
                )
            else:
                raise SonfileException(
                    f"{self.filename} is not a valid file, though it does contain the correct extension. \n"
                    f"Double check the file contains valid data."
                )

    def __repr__(self):
        return self._formatted_info()

    def __str__(self):
        return self._formatted_info()

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, log_level):
        self._log_level = log_level
        self._logger.setLevel(log_level)

    @logger.setter
    def logger(self, new_logger):
        self._logger = new_logger
        self._logger.setLevel(self._log_level)

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        self._savedir = savedir

    def _formatted_info(self):
        nchan = len(self.waves)  # Assuming 'waves' is a dataframe or similar
        nevents = len(self.events)  # Assuming 'events' is a dataframe or similar

        waves_info = self.waves.to_string(index=False)  # Formatting dataframe as string
        events_info = self.events.to_string(
            index=False
        )  # Formatting dataframe as string

        info = (
            f"{self.filename} | nchan = {nchan} | nevents = {nevents}\n"
            "-----\n"
            "waves\n"
            "-----\n"
            f"{waves_info}\n"
            "------\n"
            "events\n"
            "------\n"
            f"{events_info}"
        )
        return info

    def extract(self, *args):
        """Extract "waves" or "events" into hdf5 cache."""
        new5 = spk_io.create_empty_data_h5(self.savedir, True)
        spk_io.create_hdf_arrays(new5, self.units, self.lfps, self.events)
        for k in args:
            if k == "events":
                for _, row in self.events.iterrows():
                    self._process_event(
                        row,
                    )
            elif k == "waves":
                with tables.open_file(new5, "a") as hf5:
                    x = 1
                    do_time = True
                    if hf5.root.raw_time.time_vector.size_on_disk != 0:
                        do_time = False
                    for _, row in self.waves.iterrows():
                        worked = self._process_signal(row, hf5, do_time)
                        if worked:
                            do_time = False
                    hf5.flush()
            else:
                raise ValueError(f"Invalid argument {k}.")

    def _process_event(self, row):
        """
        Process event channels, i.e. channels that contain events.

        Event times are converted to seconds.
        """
        idx, name, chantype, fs, units = row
        self.logger.info(
            f"Processing event:"
            f"idx={idx}, name={name}, chantype={chantype}, fs={fs}, units={units}"
        )

        try:
            marks = self.sonfile.ReadMarkers(idx, int(2e9), 0)

            # convert the char ascii-encoded ints to a string
            char_codes = [
                codes_to_string([mark.Code1, mark.Code2, mark.Code3, mark.Code4])
                for mark in marks
            ]

            # create a boolean mask for filtering both char_codes and ticks
            is_printable_mask = [
                all(char in string.printable for char in code) for code in char_codes
            ]

            # filter char_codes and ticks based on the boolean mask
            filtered_codes = list(compress(char_codes, is_printable_mask))
            ticks = [mark.Tick for mark in marks]
            filtered_ticks = np.array(list(compress(ticks, is_printable_mask)))

            # convert the filtered clock ticks to seconds
            event_time = np.round(ticks_to_time(filtered_ticks, self._time_base()), 3)
            events = np.vstack((filtered_codes, event_time)).T

            return filtered_codes, event_time

        # better to not guess what error sonpy is going to throw
        except Exception as e:
            self.errors[f"{idx}_{name}_{chantype}"] = e
            self.logger.warning(f"Error reading marker:{name},{chantype}, {e}")
            return [], []

    def read_data_in_chunks(self, channel_index):
        item_size = self.sonfile.ItemSize(channel_index)
        total_bytes = self.sonfile.ChannelBytes(channel_index)
        total_items = total_bytes // item_size
        chunk_size = 1000000

        start_idx = 0

        while start_idx < total_items:
            end_idx = min(start_idx + chunk_size, total_items)
            num_items = end_idx - start_idx
            chunk_data = self.sonfile.ReadFloats(channel_index, num_items, start_idx)
            yield chunk_data
            start_idx = end_idx

    def _process_signal(self, row, hf5, do_time):
        chans = [node._v_name for node in hf5.root.raw_unit._f_list_nodes()]
        if row["name"] not in chans:
            return False
        try:
            idx, channel_name, chantype, fs, units = row
            self.logger.info(
                f"Processing waveform: idx={idx}, name={channel_name}, chantype={chantype}, fs={fs}, units={units}"
            )

            start_time = 0
            for chunk in self.read_data_in_chunks(idx):
                # turn chunk into hf5 array
                hf5.root.raw_unit[channel_name].append(np.array(chunk))
                # attach fs to attr
                hf5.root.raw_unit[channel_name]._v_attrs.fs = fs
                if do_time:
                    time_vector = np.arange(
                        start_time, start_time + len(chunk) / fs, 1 / fs
                    )
                    hf5.root.raw_time.time_vector.append(time_vector)
                    start_time += len(chunk) / fs
            hf5.flush()
            return True

        except SonfileException as e:
            self.errors["ReadFloats"] = e
            self.logger.error(f"Error reading floats: {e}")
            return False

    def save(self, filepath: str | Path, overwrite_existing=True) -> Path:
        """
        Save the data to a h5 file in the users h5 directory.

        The resulting h5 file will be saved to the data/h5 directory with the corresponding filename.
        The h5 file will contain:

        1) Metadata specific to the file, such as the bandpass filter frequencies.
            - 'metadata_file'
        2) Metadata specific to each channel, such as the channel type and sampling rate.
            - 'metadata_channel'
        3) The actual data, i.e. the waveforms and spike times.
            - 'data'

        Parameters
        ----------
        filepath : str or Path
            The path/filename including ext to save to.
        overwrite_existing : bool, optional
            Whether to overwrite an existing file with the same name. Default is False.

        Returns
        -------
        filename : Path
            The filename that the data was saved to.

        """

        filepath = Path(filepath)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")
        if not filepath.parent.exists():
            self.logger.info(f"Creating {filepath.parent} directory.")
            filepath.parent.mkdir(exist_ok=True, parents=True)
        if filepath.exists():
            if overwrite_existing:
                self.logger.info(f"Overwriting existing file: {filepath}")
                spk_io.save_channel_h5(
                    str(filepath), "channels", self.channels, self.metadata
                )
                self.logger.info(f"Saved data to {filepath}")
                return self.filename
            else:  # don't overwrite existing file
                self.logger.info(
                    f"{filepath} already exists. Set overwrite_existing=True to overwrite. Skipping h5 write."
                )
                pass
        else:
            self.logger.info(f"Saving data to {filepath}")
            spk_io.save_channel_h5(
                str(filepath), "channels", self.channels, self.metadata
            )
        return self.filename

    # wrappers for sonfile methods with additional information
    def _channel_interval(self, channel: int):
        """
        Get the waveform sample interval, in clock ticks.

        ADC channels sample waveforms in equally spaced intervals. This represents the interval at which
        that sampling occurs, and can be used to extract the sample period and thus the sampling frequency.

        """
        return self.sonfile.ChannelDivide(channel)

    def _channel_sample_period(self, channel: int):
        """
        Get the waveform sample period, in seconds.

        Takes the number of clock ticks in between each sample and divides by the base time that each tick represents.

        Parameters
        ----------
        channel : int
            The channel number.

        Returns
        -------
        sample : float
            The waveform sample interval, in seconds.

        """
        return self._channel_interval(channel) / self._time_base

    def _channel_num_ticks(self, channel: int):
        """
        Get the number of clock ticks in the channel.

        Parameters
        ----------
        channel : int
            The channel number.

        Returns
        -------
        ticks : float
            The number of clock ticks.

        """
        return self._max_time() / self._channel_sample_period(channel)

    def _channel_max_ticks(self, channel: int):
        """
        Get the last time-point in the array, in ticks.

        Parameters
        ----------
        channel : int
            The channel number.

        Returns
        -------
        last_time : float
            The last clock tick in the array.

        """
        return self.sonfile.ChannelMaxTime(channel)

    def _channel_max_time(self, channel: int):
        """
        Get the last time-point in the channel-array, in seconds.

        """
        return self._channel_max_ticks(channel) * self._time_base()

    def _time_base(self):
        """
        The number of seconds per clock tick.

        Everything in the file is quantified by the underlying clock tick (64-bit).
        All values in the file are stored, set and returned in ticks.
        You need to read this value to interpret times in seconds.

        Returns
        -------
        float
            The time base, in seconds.
        """
        return self.sonfile.GetTimeBase()

    def _max_ticks(self):
        """
        The total number of clock ticks in the file.
        """
        return self.sonfile.MaxTime()

    def _max_time(self):
        """
        The total recording length, in seconds.

        """
        return self._max_ticks() * self._time_base()

    def _max_channels(self):
        """
        The number of channels in the file.
        """
        return self.sonfile.MaxChannels()


if __name__ == "__main__":
    path_test = Path("/media/thom/hub/data/serotonin/raw/")
    animal = list(path_test.glob("*.smr"))[0]
    save_spike2_path = Path().home() / "cpl_extract"
    data = Spike2Data(animal, savepath=save_spike2_path)
    data.extract("waves")
    print("DONE")

    # data.extract("events")
    # data.save(save_test / str(data), overwrite_existing=True)
