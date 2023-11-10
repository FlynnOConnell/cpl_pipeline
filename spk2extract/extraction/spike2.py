"""
Spike data extraction utility, the main workhorse of this package.
"""
from __future__ import annotations

import string
from itertools import compress
from pathlib import Path

import numpy as np

try:
    from sonpy import lib as sp
except ImportError:
    try:
        import sonpy as sp
    except:
        pass
        # raise Warning("sonpy not found. Are you on a M1 Mac?")

from spk2extract.logs import logger
from spk2extract.defaults import defaults
from spk2extract import spk_io


class Channel:
    def __init__(self, name, chan_type, data, times, metadata):
        self.name = name
        self.type = chan_type
        self.data: np.ndarray = data
        self.times: np.ndarray = times
        self.metadata = metadata

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
    >>> from spk2extract.extraction import Spike2Data
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

    def __init__(
        self,
        filepath: Path | str,
    ):
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
        self.logger = logger
        self.errors = {}
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.channels = []

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
        self.bitrate = 32 if self.sonfile.is32file() else 64
        self.metadata = {
            "bitrate": self.bitrate,
            "filename": self.filename.stem,
            "recording_length": self.max_time(),
        }

    def __repr__(self):

        return f"{self.filename.stem}"

    def __str__(self):
        """Allows us to use str(spike_data.SpikeData(file)) to get the filename stem."""
        return f"{self.filename.stem}"

    def process_channels(self, ):
        """
        Iterate over the channels in the file.

        Yields
        -------
        idx : int
            The channel index.
        title : str
            The channel title.
        type : int
            The channel type.
        """
        for idx in range(self.max_channels()):
            # Get the sampling frequency of the channel
            fs = np.round(1 / (self.sonfile.ChannelDivide(idx) * self.time_base()), 2)
            metadata = {
                "fs": fs,
                "units": self.sonfile.GetChannelUnits(idx),
            }  # units will be empty for events
            signal, times = [], []
            channel_type = ""
            if self.sonfile.ChannelType(idx) == sp.DataType.Off:
                # contains no data, untitled channels
                continue
            if self.sonfile.ChannelType(idx) == sp.DataType.Adc:
                # contains analog - digital converted data
                signal = self.sonfile.ReadFloats(idx, int(2e9), 0)
                times = indices_to_time(range(len(signal)), fs)
                channel_type = "signal"
            if self.sonfile.ChannelType(idx) in (sp.DataType.Marker, sp.DataType.RealMark, sp.DataType.TextMark, sp.DataType.AdcMark):
                channel_type = "event"
                signal, times = self.process_event(idx)
            self.channels.append(
                Channel(
                    self.sonfile.GetChannelTitle(idx),
                    channel_type,
                    signal,
                    times,
                    metadata,
                )
            )

    def process_event(self, idx: int):
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
            filtered_ticks = list(compress(ticks, is_printable_mask))

            # convert the filtered clock ticks to seconds
            event_time = np.round(ticks_to_time(filtered_ticks, self.time_base()), 3)

            return filtered_codes, event_time

        except SonfileException as e:
            self.errors["ReadMarker"] = e
            self.logger.error(f"Error reading marker: {e}")
            return [], []

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
            logger.info(f"Creating {filepath.parent} directory.")
            filepath.parent.mkdir(exist_ok=True, parents=True)
        if filepath.exists():
            if overwrite_existing:
                logger.info(f"Overwriting existing file: {filepath}")
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
            logger.info(f"Saving data to {filepath}")
            spk_io.save_channel_h5(
                str(filepath), "channels", self.channels, self.metadata
            )
        return self.filename

    def channel_interval(self, channel: int):
        """
        Get the waveform sample interval, in clock ticks.

        ADC channels sample waveforms in equally spaced intervals. This represents the interval at which
        that sampling occurs, and can be used to extract the sample period and thus the sampling frequency.

        """
        return self.sonfile.ChannelDivide(channel)

    def channel_sample_period(self, channel: int):
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
        return self.channel_interval(channel) / self.time_base

    def channel_num_ticks(self, channel: int):
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
        return self.max_time() / self.channel_sample_period(channel)

    def channel_max_ticks(self, channel: int):
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

    def channel_max_time(self, channel: int):
        """
        Get the last time-point in the channel-array, in seconds.

        """
        return self.channel_max_ticks(channel) * self.time_base()

    def time_base(self):
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

    def max_ticks(self):
        """
        The total number of clock ticks in the file.
        """
        return self.sonfile.MaxTime()

    def max_time(self):
        """
        The total recording length, in seconds.

        """
        return self.max_ticks() * self.time_base()

    def max_channels(self):
        """
        The number of channels in the file.

        """
        return self.sonfile.MaxChannels()


if __name__ == "__main__":
    defaults = defaults()
    log_level = defaults["log_level"]
    logger.setLevel(log_level)
    print(f"Log level set to {log_level}")

    path_test = Path().home() / "data" / "serotonin"
    save_test = Path().home() / "data" / "extracted" / "serotonin"

    # iterate over each folder in path_test
    errors = []
    for animal_path in path_test.iterdir():
        save_test = save_test / animal_path.stem

        if not animal_path.is_dir():
            continue

        # iterate over each file in folder
        for file in animal_path.glob("*.smr"):
            filename = file.stem
            try:
                data = Spike2Data(file)
                data.process_channels()
                data.save(save_test / str(data), overwrite_existing=True)
            except Exception as e:
                logger.info(f"Error processing {file}: {e}")
                errors.append(e)
                continue