"""
Spike data extraction utility, the main workhorse of this package.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sonpy import lib as sp

import spk2extract
from spk2extract.spk_log.logger_config import configure_logger
from spk2extract.util import extract_waveforms, filter_signal
from spk2extract.util.cluster import detect_spikes

UnitData = namedtuple("UnitData", ["spikes", "times"])

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


# Function to load a SpikeData instance from a h5 file
def __load_spk2_from_h5(filename: str | Path):
    """
    Load data from h5.
    """
    spike_data = {"metadata": {}, "sampling_rates": {}, "unit": {}}
    with h5py.File(filename, "r") as f:
        for key, value in f["metadata"].attrs.items():
            spike_data["metadata"][key] = value

        for key, value in f["sampling_rates"].attrs.items():
            spike_data["sampling_rates"][key] = value

        for channel in f["unit"].keys():
            spikes = np.array(f["unit"][channel]["spikes"])
            times = np.array(f["unit"][channel]["times"])
            spike_data["unit"][channel] = UnitData(spikes, times)
    return spike_data


# Function to merge two SpikeData instances
def __merge_spk2(spike_data1, spike_data2):
    """
    Merge spike data.
    """
    merged_spike_data = {"metadata": spike_data1["metadata"], "unit": {}}

    # Merge unit data
    for channel in spike_data1["unit"].keys():
        spikes1, times1 = spike_data1["unit"][channel]
        spikes2, times2 = spike_data2["unit"][channel]

        merged_spikes = np.concatenate([spikes1, spikes2])
        merged_times = np.concatenate([times1, times2])

        merged_spike_data["unit"][channel] = UnitData(merged_spikes, merged_times)
    return merged_spike_data


def __merge_spk2_from_dict(spike_data_dict1, spike_data_dict2):
    """
    Merge data from dicts.
    """
    merged_spike_data = {
        "metadata": spike_data_dict1["metadata"],
        "sampling_rates": spike_data_dict1["sampling_rates"],
        "unit": {},
    }

    # find which spikedata has preinfusion data
    if spike_data_dict1["metadata"]["infusion"] == "pre":
        spike_data_pre = spike_data_dict1
        spike_data_post = spike_data_dict2
    else:
        spike_data_pre = spike_data_dict2
        spike_data_post = spike_data_dict1

    # Merge unit data
    for channel in spike_data_pre["unit"].keys():
        spikes1 = spike_data_pre["unit"][channel].spikes
        times1 = spike_data_pre["unit"][channel].times
        spikes2 = spike_data_post["unit"][channel].spikes
        times2 = spike_data_post["unit"][channel].times

        merged_spikes = np.concatenate([spikes1, spikes2])
        merged_times = np.concatenate([times1, times2])

        merged_spike_data["unit"][channel] = UnitData(merged_spikes, merged_times)

    return merged_spike_data


def write_complex_h5(
    filename: Path | str,
    metadata_file: dict,
    metadata_channel: dict[Any],
    data: dict,
):
    """
    Creates a h5 file specific to the spike2 dataset.

    There is a group for metadata, metadata_dict, data, and sampling_rates.

    .. note::

        The metadata_file group contains metadata for the entire file, i.e. the bandpass filter frequencies.
        The metadata_channel group contains metadata for each channel, i.e. the channel type, sampling rate.
        Data contains the actual data, i.e. the waveforms and spike times.

    Parameters
    ----------
    filename : str or Path
        The filename to save to.
    metadata_file : dict
        A dictionary containing the simple str metadata.
    metadata_channel : dict
        A dictionary containing the metadata dictionaries.
    data : dict
        A dictionary containing the data.

    Returns
    -------
    None

    """
    with h5py.File(filename, "w") as f:
        # Save metadata (any simple type supported by HDF5)
        metadata_file_grp = f.create_group("metadata_file")
        for key, value in metadata_file.items():
            metadata_file_grp.attrs[key] = value

        # Save metadata dictionaries
        metadata_channel_grp = f.create_group("metadata_channel")
        for dict_name, dict_data in metadata_channel.items():
            sub_group = metadata_channel_grp.create_group(dict_name)
            for key, value in dict_data.items():
                str_key = (
                    str(key) if not isinstance(key, tuple) else ",".join(map(str, key))
                )
                sub_group.attrs[
                    str_key
                ] = value  # this value can be int, float, str, etc.

        # Save data dictionaries
        data_grp = f.create_group("data")
        for data_key, data_value in data.items():
            sub_group = data_grp.create_group(data_key)
            # Convert the namedtuple to a dict to store in the h5 file as a subgroup
            # The leading _ in _asdict() is to prevent named conflicts, not to indicate privacy
            # See: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict
            for field_name, field_value in data_value._asdict().items():
                sub_group.create_dataset(field_name, data=np.array(field_value))
    return None

class SpikeData:
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
    >>> from spk2extract import SpikeData
    >>> from pathlib import Path
    >>> smr_path = Path().home() / "data" / "smr"
    >>> files = [file for file in smr_path.glob("*.smr")]
    >>> assert len(files) > 0
    >>> data = SpikeData(files[0])
    >>> data
    'rat1-2021-03-24_0001'
    >>> data.extract()
    >>> data["LFP1"]
    UnitData(spikes=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00],), times=array([...]))

    >>> data["LFP1"].spikes
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,]])
    >>> data["LFP1"].times
    array([0.00000000e+00,  0.00000000e+00,  0.00000000e+00,])
    >>> data["LFP1"].spikes.shape
    (1, 3)

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
        self.logger = configure_logger(__name__, spk2extract.log_dir, level=logging.DEBUG)
        self._bandpass_low = 300
        self._bandpass_high = 3000
        self.errors = {}
        self.exclude = exclude
        self.empty = False
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        if self.sonfile.GetOpenError() != 0:
            if self.filename.suffix != ".smr":
                raise SonfileException(f"{self.filename} is not a valid file. \n"
                                       f"Extension {self.filename.suffix} is not valid.")
            else:
                raise SonfileException(f"{self.filename} is not a valid file, though it does contain the correct extension. \n"
                                       f"Double check the file contains valid data.")
        self.bitrate = 32 if self.sonfile.is32file() else 64
        self.metadata_channel = {}
        self.data = {}
        self.metadata_file = self.bundle_metadata()

    def __repr__(self):
        return f"{self.filename.stem}"

    def __str__(self):
        """Allows us to use str(spike_data.SpikeData(file)) to get the filename stem."""
        return f"{self.filename.stem}"

    def __bool__(self):
        return self.empty

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError(f"{key} not found in SpikeData object.")

    def extract(
        self,
    ):
        """
        Main workhorse of this class; extracts unit data from the Spike2 file.

        This function uses the sonpy library to extract unit data from the Spike2 file. It converts all the
        intrinsic data stored in the spike2 file into convenient Python objects, such as dictionaries, namedtuples
        and properties. SonPy is a python wrapper for the CED C++ library, the sonpy library is not well documented
        and the CED C++ library is not open source.
        """
        self.logger.debug(f"Extracting ADC channels from {self.filename.stem}")

        for idx in range(self.max_channels):
            title = self.sonfile.GetChannelTitle(idx)
            if (
                self.sonfile.ChannelType(idx) == sp.DataType.Adc
                and title not in self.exclude
                and "LFP" not in title
            ):
                self.logger.debug(f"Processing {title}")
                sampling_rate = np.round(
                    1 / (self.sonfile.ChannelDivide(idx) * self.time_base), 2
                )

                # noinspection PyArgumentList
                waveforms = self.sonfile.ReadFloats(idx, int(2e9), 0)

                # Ensure the Nyquist-Shannon sampling theorem is satisfied
                if sampling_rate < (2 * self.bandpass_high):
                    raise ValueError(
                        "Sampling rate is too low for the given bandpass filter frequencies."
                    )

                # Low/high bandpass filter
                filtered_segment = filter_signal(
                    waveforms,
                    sampling_rate,
                    (self.bandpass_low, self.bandpass_high),
                )

                # Extract spikes and times from the filtered segment
                slices, spike_times, thresh = detect_spikes(
                    filtered_segment,
                    [0.5, 1.0],
                    sampling_rate,
                    # spike_snapshot,
                )
                slices = np.array(slices)
                spike_times = np.array(spike_times)

                # Create a FinalUnitData namedtuple with the concatenated spikes and times
                final_unit_data = UnitData(spikes=slices, times=spike_times)

                # Store this namedtuple in the self.unit dictionary
                self.data[title] = final_unit_data
                self.metadata_channel[title] = {
                    "sampling_rate": sampling_rate,
                    "channel_type": "unit",
                    "channel_number": idx + 1,
                    "channel_title": title,
                    "channel_units": self.sonfile.GetChannelUnits(idx),
                    "channel_divide": self.sonfile.ChannelDivide(idx),
                    "channel_interval": self.channel_interval(idx),
                    "channel_sample_period": self.channel_sample_period(idx),
                    "channel_num_ticks": self.channel_num_ticks(idx),
                    "channel_max_ticks": self.channel_max_ticks(idx),
                    "channel_max_time": self.channel_max_time(idx),
                }

        self.logger.debug(f"Finished extracting ADC channels from {self.filename.stem}")

    def save(self, overwrite_existing=False) -> Path:
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
        overwrite_existing : bool, optional
            Whether to overwrite an existing file with the same name. Default is False.

        Returns
        -------
        filename : Path
            The filename that the data was saved to.

        """
        from spk2extract import spk2dir # prevent circular import
        h5path = spk2dir / "h5" / self.filename.stem
        h5path = h5path.with_suffix(".h5")
        if not h5path.parent.exists():
            h5path.parent.mkdir(exist_ok=True, parents=True)
        if h5path.exists() and not overwrite_existing:
            self.logger.info(f"{h5path} already exists. Set overwrite_existing=True to overwrite. Skipping h5 write.")
            pass
        try:
            write_complex_h5(h5path, self.metadata_file, self.metadata_channel, self.data,)
            self.logger.info(f"Saved data to {h5path}")
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
        return self.max_time / self.channel_sample_period(channel)

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
        return self.channel_max_ticks(channel) * self.time_base

    @property
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

    @property
    def max_ticks(self):
        """
        The total number of clock ticks in the file.
        """
        return self.sonfile.MaxTime()

    @property
    def max_time(self):
        """
        The total recording length, in seconds.

        """
        return self.max_ticks * self.time_base

    @property
    def max_channels(self):
        """
        The number of channels in the file.

        """
        return self.sonfile.MaxChannels()

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

    def bundle_metadata(self):
        """
        Bundle the metadata into a dictionary.

        Returns
        -------
        metadata : dict
            A dictionary containing the metadata.

        """
        return {
            "bandpass": [self.bandpass_low, self.bandpass_high],
            "time_base": self.time_base,
            "max_time": self.max_ticks,
            "recording_length": self.max_time,
            "filename": self.filename.stem,
            "exclude": self.exclude,
        }


if __name__ == "__main__":
    path_test = Path().home() / "data" / "smr"
    test_files = [file for file in path_test.glob("*.smr")]
    for testfile in test_files:
        testdata = SpikeData(
            testfile,
            ("Respirat", "RefBrain", "Sniff"),
        )
        testdata.extract()
        testdata.save(overwrite_existing=True)