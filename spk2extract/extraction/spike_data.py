"""
==========
Spike2Data
==========

.. currentmodule:: extraction

"""
from __future__ import annotations

import logging
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
from sonpy import lib as sp

from spk2extract.logging.logger_config import configure_logger
from spk2extract.util import extract_waveforms, filter_signal

logfile = Path().home() / "data" / "spike_data.log"
logger = configure_logger(__name__, logfile, level=logging.DEBUG)

UnitData = namedtuple("UnitData", ["spikes", "times"])


def __get_base_filename__(
    name_data: SpikeData | dict,
) -> str:
    """
    Extract the base filename from a SpikeData object.
    """
    return (
        name_data["metadata"]["filename"]
        .replace("_preinfusion", "")
        .replace("_postinfusion", "")
    )


def __save_spikedata_to_h5(spike_data: SpikeData, filename: str | Path) -> None:
    """
    Save data to h5 file.

    Parameters
    ----------
    spike_data : SpikeData
        The SpikeData instance to save.
    filename : str or Path
        The filename to save to.

    Returns
    -------
    None
    """
    with h5py.File(filename, "w") as f:
        metadata_grp = f.create_group("metadata")
        for key, value in spike_data.metadata.items():
            metadata_grp.attrs[key] = value

        sampling_rate_group = f.create_group("sampling_rates")
        for channel, sample_rate in spike_data.sampling_rates.items():
            sampling_rate_group.attrs[channel] = sample_rate

        unit_grp = f.create_group("unit")
        for channel, unit_data in spike_data.unit.items():
            channel_grp = unit_grp.create_group(channel)
            channel_grp.create_dataset("spikes", data=unit_data.spikes)
            channel_grp.create_dataset("times", data=unit_data.times)
    logger.debug(f"Saved data successfully to {filename}")


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


def __save_merged_to_h5(merged_spike_data: dict, filename: str | Path):
    """
    Save to h5 merge.
    """
    with h5py.File(filename, "w") as f:
        metadata_grp = f.create_group("metadata")
        for key, value in merged_spike_data["metadata"].items():
            metadata_grp.attrs[key] = value

        unit_grp = f.create_group("unit")
        for channel, unit_data in merged_spike_data["unit"].items():
            channel_grp = unit_grp.create_group(channel)
            channel_grp.create_dataset("spikes", data=unit_data.spikes)
            channel_grp.create_dataset("times", data=unit_data.times)
    logger.debug(f"Saved merged data successfully to {filename}")


class SpikeData:
    """
    Container class for Spike2 data.

    Can be used as:
    - A dictionary, where the keys are the channel names and the values are the waveforms (LFP + Unit).
    - A list, where the elements are the channel names.
    - A boolean, where True means the file is empty and False means it is not.
    - A string, where the string is the filename stem.

    """

    @staticmethod
    def __validate_same_metadata__(meta1, meta2):
        for key in meta1:
            if key not in ["filename", "infusion", "max_time", "recording_length"]:
                if meta1[key] != meta2[key]:
                    return False
        return True

    def __init__(
        self,
        filepath: Path | str,
        exclude: tuple = ("Respirat", "Sniff", "RefBrain"),
    ):
        """
        Class for reading and storing data from a Spike2 file.

        .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

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
        self._bandpass_low = 300
        self._bandpass_high = 3000
        self.errors = {}
        self.exclude = exclude
        self.empty = False
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.bitrate = 32 if self.sonfile.is32file() else 64
        self.unit = {}
        self.sampling_rates = {}
        self.metadata = self.bundle_metadata()
        self.process_units()

    def __repr__(self):
        return f"{self.filename.stem}"

    def __str__(self):
        """Allows us to use str(spike_data.SpikeData(file)) to get the filename stem."""
        return f"{self.filename.stem}"

    def __bool__(self):
        return self.empty

    def __getitem__(self, key):
        if key in self.unit:
            return self.unit[key]
        else:
            raise KeyError(f"{key} not found in SpikeData object.")

    def process_units(
        self,
    ):
        """
        Main workhorse of this package; extracts unit data from the Spike2 file.

        This function uses the sonpy library to extract unit data from the Spike2 file. It converts all the
        intrinsic data stored in the spike2 file into convenient Python objects, such as dictionaries, namedtuples
        and properties. SonPy is a python wrapper for the CED C++ library, the sonpy library is not well documented
        and the CED C++ library is not open source.
        """
        logger.debug(f"Extracting ADC channels from {self.filename.stem}")

        for idx in range(self.max_channels):
            title = self.sonfile.GetChannelTitle(idx)
            if (
                self.sonfile.ChannelType(idx) == sp.DataType.Adc
                and title not in self.exclude
                and "LFP" not in title
            ):
                logger.debug(f"Processing {title}")
                sampling_rate = np.round(
                    1 / (self.sonfile.ChannelDivide(idx) * self.time_base), 2
                )
                self.sampling_rates[title] = sampling_rate

                # Extract and filter waveforms for this chunk
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
                slices, spike_times = extract_waveforms(
                    filtered_segment,
                    sampling_rate,
                )

                # Dejitter the spike times
                # slices, spike_times = dejitter(spike_times, sampling_rate, sampling_rate)

                # Create a FinalUnitData namedtuple with the concatenated spikes and times
                final_unit_data = UnitData(spikes=slices, times=spike_times)

                # Store this namedtuple in the self.unit dictionary
                self.unit[title] = final_unit_data

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
            "infusion": "pre" if self.preinfusion else "post",
            "filename": self.filename.stem,
            "exclude": self.exclude,
        }


if __name__ == "__main__":
    path_test = Path().home() / "data" / "smr"
    path_combined = Path().home() / "data" / "combined"
    path_combined.mkdir(exist_ok=True, parents=True)
    files = [f for f in path_test.glob("*.h5")]

    # # load the h5
    data = []
    for file in files:
        data.append(__load_spk2_from_h5(file))

    # merge the data
    merged_data = __merge_spk2_from_dict(data[0], data[1])

    # save the merged data
    basename = __get_base_filename__(merged_data)
    fname = path_combined / (basename + ".h5")
    __save_merged_to_h5(merged_data, fname)
    # for file in files:
    #     data = SpikeData(
    #         file,
    #         ("Respirat", "RefBrain", "Sniff"),
    #     )
    #     save_spike_data_to_h5(data, file.with_suffix(".h5"))
    x = 5
