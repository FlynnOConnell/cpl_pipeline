"""
Spike data extraction utility, the main workhorse of this package.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np

try:
    from sonpy import lib as sp
except ImportError:
    try:
        import sonpy as sp
    except:
        raise ImportError("sonpy not found. Are you on a M1 Mac?")

from spk2extract.logs import logger
from spk2extract.utils import check_substring_content
from spk2extract.defaults import defaults
from spk2extract import spk_io
from spk2extract.util import filter_signal
from spk2extract.util.cluster import detect_spikes

WaveData = namedtuple("WaveData", ["spikes", "times"])
EventData = namedtuple("EventData", ["events", "times"])


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


def indices_to_time(indices, fs):
    """Spike2 indices are in clock ticks, this converts them to seconds."""
    return np.array(indices) / float(fs)


def ticks_to_time(ticks, time_base):
    """Converts clock ticks to seconds."""
    return np.array(ticks) * time_base


def is_ascii_letter(char):
    if char in range(65, 91) or char in range(97, 123):
        return True


def codes_to_string(codes):
    return "".join(chr(code) for code in codes if code != 0)


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
    >>> data.get_waves()
    >>> data["LFP1"]
    WaveData(spikes=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00],), times=array([...]))

    >>> data["LFP1"].spikes
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,]])
    >>> data["LFP1"].times
    array([0.00000000e+00,  0.00000000e+00,  0.00000000e+00,])

    """

    def __init__(
        self,
        filepath: Path | str,
        exclude: tuple = ("Respirat", "Sniff", "RefBrain"),
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
        self._bandpass_low = 300
        self._bandpass_high = 3000
        self.errors = {}  # errors that won't stop the extraction
        self.exclude = exclude  # channels to exclude from the extraction
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.events = None
        if self.sonfile.GetOpenError() != 0:
            if self.filename.suffix != ".smr":
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
        self.metadata_channel = {}  # filled in during get_waves()
        self.data = {}
        self.metadata_file = {
            "bitrate": self.bitrate,
            "filename": self.filename.stem,
            "recording_length": self.max_time(),
        }

    def __repr__(self):
        return f"{self.filename.stem}"

    def __str__(self):
        """Allows us to use str(spike_data.SpikeData(file)) to get the filename stem."""
        return f"{self.filename.stem}"

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError(f"{key} not found in SpikeData object.")

    @property
    def bandpass_low(self):
        """
        The lower bound of the bandpass filter.
        """
        return self._bandpass_low

    @bandpass_low.setter
    def bandpass_low(self, value):
        """
        Set the lower bound of the bandpass filter.

        Parameters
        ----------
        value : int
            The lower bound of the bandpass filter.

        Returns
        -------
            None
        """
        self._bandpass_low = value

    @property
    def bandpass_high(self):
        """
        The upper bound of the bandpass filter.

        Returns
        -------
            int : The upper bound of the bandpass filter.

        """
        return self._bandpass_high

    @bandpass_high.setter
    def bandpass_high(self, value):
        """
        Set the upper bound of the bandpass filter.

        Parameters
        ----------
        value: int
            The upper bound of the bandpass filter.

        Returns
        -------
        None

        """
        self._bandpass_high = value

    def get_events(self):
        logger.info("Extracting events...")
        for idx in range(self.max_channels()):
            title = (self.sonfile.GetChannelTitle(idx)).lower()
            if "keyboard" in title:
                try:
                    # noinspection PyArgumentList
                    marks = self.sonfile.ReadMarkers(idx, int(2e9), 0)

                    # Get string representation of the spike2 codes and corresponding times
                    char_codes = [
                        codes_to_string(
                            [mark.Code1, mark.Code2, mark.Code3, mark.Code4]
                        )
                        for mark in marks
                    ]
                    time_conv = np.round(
                        ticks_to_time([mark.Tick for mark in marks], self.time_base()),
                        3,
                    )
                    self.events = EventData(events=char_codes, times=time_conv)
                except SonfileException as e:
                    self.errors["ReadMarker"] = e

    def get_waves(
        self,
    ):
        """
        Main workhorse of this class; extracts unit and lfp data from the Spike2 file.

        This function uses the sonpy library to extract unit data from the Spike2 file. It converts all the
        intrinsic data stored in the spike2 file into convenient Python objects, such as dictionaries, namedtuples
        and properties. SonPy is a python wrapper for the CED C++ library, the sonpy library is not well documented
        and the CED C++ library is not open source.
        """
        self.logger.debug(f"-| Extracting ADC channels from {self.filename.stem} -|")
        for idx in range(self.max_channels()):
            title = self.sonfile.GetChannelTitle(idx)
            print(title)
            pass
            if (
                self.sonfile.ChannelType(idx) == sp.DataType.Adc
                and title not in self.exclude
            ):
                self.logger.debug(f"Processing {title}")
                fs = np.round(
                    1 / (self.sonfile.ChannelDivide(idx) * self.time_base()), 2
                )

                # Read the waveforms from the channel, up to 2e9, or 2 billion samples at a time which represents
                # the maximum amount of 30 bit floats that can be stored in memory
                # noinspection PyArgumentList
                waveforms = self.sonfile.ReadFloats(idx, int(2e9), 0)

                # Ensure the Nyquist-Shannon sampling theorem is satisfied
                if fs < (2 * self.bandpass_high):
                    # TODO: handle this
                    pass

                chan_type = None
                applied_filter = None
                # Titles for unit are often just U, but we need to make sure we don't include LFP channels
                # with a U in the title anywhere
                if check_substring_content(title, "u") and not check_substring_content(
                    title, "lfp"
                ):
                    chan_type = "unit"
                    filtered_segment = filter_signal(
                        waveforms,
                        fs,
                        (self.bandpass_low, self.bandpass_high),
                    )
                    applied_filter = (self.bandpass_low, self.bandpass_high)

                    # Extract spikes and times from the filtered segment
                    slices, spike_indices, thresh = detect_spikes(
                        filtered_segment,
                        (0.5, 1.0),
                        fs,
                    )
                    slices = np.array(slices)
                    spike_times = spike_indices / float(fs)

                    # Create a FinalWaveData namedtuple with the concatenated spikes and times
                    final_wave_data = WaveData(spikes=slices, times=spike_times)

                    # Store this namedtuple in the self.unit dictionary
                    self.data[title] = final_wave_data

                elif check_substring_content(title, "lfp"):
                    chan_type = "lfp"
                    filtered_segment = filter_signal(
                        waveforms, fs, (0.3, 500)  # TODO: add this as a parameter
                    )
                    applied_filter = (0.3, 500)
                    self.data[title] = WaveData(
                        spikes=filtered_segment,
                        times=np.arange(len(filtered_segment)) / float(fs),
                    )
                self.metadata_channel[title] = {
                    "fs": fs,
                    "filter": applied_filter,
                    "type": chan_type,
                    "units": self.sonfile.GetChannelUnits(idx),
                }

        self.logger.debug(f"Spike extraction complete: {self.filename.stem}")

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
                spk_io.write_h5(
                    filepath,
                    self.data,
                    self.events,
                    self.metadata_channel,
                    self.metadata_file,
                )
                self.logger.info(f"Saved data to {filepath}")
                return self.filename
            else:  # don't overwrite existing file
                self.logger.info(
                    f"{filepath} already exists. Set overwrite_existing=True to overwrite. Skipping h5 write."
                )
                pass
        try:
            spk_io.write_h5(
                filepath,
                self.data,
                self.events,
                self.metadata_channel,
                self.metadata_file,
            )
            self.logger.info(f"Saved data to {filepath}")
        except Exception as e:
            self.logger.error(f"Error writing h5 file: {e}")
            raise e
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

    def bundle_metadata(self):
        """
        Bundle the metadata into a dictionary.

        Returns
        -------
        metadata : dict
            A dictionary containing the metadata.

        """
        return {
            "filename": self.filename.stem,
            "recording_length": self.max_time(),
        }


if __name__ == "__main__":
    defaults = defaults()
    log_level = defaults["log_level"]
    logger.setLevel(log_level)
    print(f"Log level set to {log_level}")

    path_test = Path().home() / "data" / "context" / 'dk1'
    save_test = Path().home() / "data" / "extracted" / 'dk1'
    save_test.mkdir(exist_ok=True, parents=True)
    test_files = [file for file in path_test.glob("*.smr")]
    for testfile in test_files:
        testdata = Spike2Data(
            testfile,
        )
        testdata.get_events()
        testdata.get_waves()
        testdata.save(save_test / str(testdata), overwrite_existing=True)
