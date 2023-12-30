from __future__ import annotations

import string
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd

from cpl_pipeline.utils import check_substring_content

try:
    from sonpy import lib as sp
except ImportError:
    try:
        import sonpy as sp
    except:
        pass

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

SIGNALS = [sp.DataType.Adc]


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
    Spike2 CED Software data extractor. This class is used to extract data from a Spike2 file.
    Both 32bit and 64bit files (.smr and .smrx) are supported.

    Parameters:
    -----------
    filepath : Path | str
        The full path to the Spike2 file, including filename + extension.
    exclude : tuple, optional
        A tuple of channel names to exclude from the data. Default is empty tuple.

    Attributes:
    -----------
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
    >>> from cpl_pipeline.extract import Spike2Data
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

    def __init__(self, filepath: Path | str):
        """
        Wrapper class for reading and storing data from a Spike2 file.

        Parameters:
        -----------
        filepath : Path | str
            The full path to the Spike2 file, including filename + extension.

        Attributes:
        -----------
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

        self.filename = Path(filepath)
        self._loaded = False

        self.data = pd.DataFrame()

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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpickleable entries.
        del state["_sonfile"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_sonfile_in_memory()

    def __repr__(self):
        return f"{self.filename.stem}"

    def __str__(self):
        # Helper function to format channel details
        def format_channel_details(channels):
            return ", ".join(
                f"{row['electrode']}:{row['name']}" for _, row in channels.iterrows()
            )

        bitrate_str = (
            f"{self.bitrate}-bit (.smr)"
            if self.bitrate == 32
            else f"{self.bitrate}-bit (.smrx)"
        )
        info = [
            f"Dataset: {self.filename.stem}",
            f"\n" f"{'Raw Filepath:':<35} {self.filename}",
            f"{'Recording Length:':<35} {self.rec_length} seconds",
            f"{'Bitrate:':<35} {bitrate_str}",
            f"{'Number of Used Channels:':<35} {len(self.data)}",
        ]

        if not self.events.empty:
            info.append(
                f"{len(self.events)}{' Event Channels (index, Type):':<35} {format_channel_details(self.events)}"
            )
        if not self.waves.empty:
            info.append(f"{'#Waveform Channels:':<35} {len(self.waves)}")
            info.append("-" * 35)
        if not self.units.empty:
            info.append(f"{'#Unit Channels:':<35} {len(self.units)}")
            info.append(f"{'Unit Channels:':<35} {format_channel_details(self.units)}")
        if not self.lfps.empty:
            info.append(f"{'Number of LFP Channels:':<35} {len(self.lfps)}")
            info.append(f"{'LFP Channels:':<35} {format_channel_details(self.lfps)}")

        return "\n".join(info)

    @property
    def sonfile(self):
        """
        The SonFile object from the sonpy library.
        """
        if self._loaded:
            return self._sonfile
        else:
            self._load_sonfile_in_memory()
            self._loaded = True
            return self._sonfile

    def read_data_in_chunks(self, channel_index, event_type, chunk_size=None):
        """
        Read data from a channel in chunks.

        Parameters
        ----------
        channel_index : int
            The channel index to read from, spike2 channels.
        event_type : str
            The data-type of event to read, either "wave" or "event".

        Returns
        -------
        generator
            A generator that yields chunks of data from the channel.
        """
        chunk_size = chunk_size if chunk_size else 32 * 1024
        item_byte_size = self.sonfile.ItemSize(channel_index)
        total_bytes = self.sonfile.ChannelBytes(channel_index)
        total_items = total_bytes // item_byte_size  # approximate number of items

        start_idx = 0

        while start_idx < total_items:
            end_idx = min(start_idx + chunk_size, total_items)
            num_items = end_idx - start_idx
            if event_type == "event":

                marks = self.sonfile.ReadMarkers(channel_index, num_items, start_idx)

                # spike2 sends a 4-byte ascii-encoded int for each character in the string
                # convert those ints to a string
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

                yield events

            elif event_type == "wave":
                chunk_data = self.sonfile.ReadFloats(channel_index, num_items, start_idx)
                yield chunk_data
            start_idx = end_idx

    def load_metadata(self):

        channel_indices = range(self._max_channels())

        self.data = pd.DataFrame(
            {
                "electrode": channel_indices,
                "name": [self.sonfile.GetChannelTitle(idx) for idx in channel_indices],
                "port": [self.sonfile.PhysicalChannel(idx) for idx in channel_indices],
                "units": [self.sonfile.GetChannelUnits(idx) for idx in channel_indices],
                "offsets": [self.sonfile.GetChannelOffset(idx) for idx in channel_indices],
                "scales": [self.sonfile.GetChannelScale(idx) for idx in channel_indices],
                "sampling_rate": [
                    np.round(
                        1 / (self.sonfile.ChannelDivide(idx) * self._time_base()), 2
                    )
                    for idx in channel_indices
                ],
                "SonpyType": [self.sonfile.ChannelType(idx) for idx in channel_indices],
            }
        )

        self.rec_length = self._max_time()
        self.bitrate = 32 if self.sonfile.is32file() else 64

        # Filter out channels with type DataType.Off
        self.data = self.data[self.data["SonpyType"] != sp.DataType.Off]

        # if data type is a signal, and the channel name contains "u", then it is a unit
        self.data["unit"] = self.data.apply(
            lambda row: row["SonpyType"] in SIGNALS
            and check_substring_content(row["name"], "u"),
            axis=1,
        )

        self.data["lfp"] = self.data.apply(
            lambda row: row["SonpyType"] in SIGNALS
            and check_substring_content(row["name"], "lfp"),
            axis=1,
        )

        self.data["event"] = self.data.apply(
            lambda row: row["SonpyType"] in EVENTS, axis=1
        )

        self.data_loaded = True
        self._flush_sonfile()
        return self.data

    # below are most wrappers for sonfile methods with additional information
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

    def _flush_sonfile(self):
        """
        Flush the sonfile object from memory.
        """
        self._sonfile.FlushSystemBuffers()
        self._loaded = False

    def _load_sonfile_in_memory(self):
        """
        Initialize the sonfile object.
        """
        self._sonfile = sp.SonFile(str(self.filename), True)
        self._loaded = True


if __name__ == "__main__":
    path_test = Path().home() / 'data' / 'r35'
    animal = list(path_test.glob("*.smr"))[0]
    save_spike2_path = Path().home() / "cpl_pipeline"
    data = Spike2Data(animal, savepath=save_spike2_path)
    data.extract("waves")
    print("DONE")

    # data.extract("events")
    # data.save(save_test / str(data), overwrite_existing=True)
