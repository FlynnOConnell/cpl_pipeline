import datetime as dt
import itertools
import os
import pprint
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from tkinter import Tk

import numpy as np
import pandas as pd
import scipy.io as sio
import tables
from icecream import ic
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

from cpl_extract import spk_io, write_dict_to_json
from cpl_extract.analysis import cluster as clust
from cpl_extract.analysis import spike_analysis
from cpl_extract.analysis.circus_interface import circus_clust
from cpl_extract.analysis.spike_analysis import make_rate_arrays
from cpl_extract.analysis.spike_sorting import make_spike_arrays, calc_units_similarity
from cpl_extract.base import objects
from cpl_extract.extract import Spike2Data
from cpl_extract.plot.data_plot import (
    plot_traces_and_outliers,
    make_unit_plots,
    plot_overlay_psth,
    plot_spike_raster,
    plot_ensemble_raster,
)
from cpl_extract.spk_io import printer as pt, writer as wt, userio, h5io
from cpl_extract.spk_io.paramio import load_params, write_params_to_json
from cpl_extract.spk_io.h5io import (
    get_h5_filename,
    write_electrode_map_to_h5,
    cleanup_clustering,
    write_digital_map_to_h5,
    write_array_to_hdf5,
    create_trial_data_table, write_spike2_array_to_h5, write_time_vector_to_h5,
)
from cpl_extract.utils.spike_sorting_GUI import launch_sorter_GUI, SpikeSorterGUI


def circus_clust_run(shell=False):
    from cpl_extract.analysis.circus_interface import circus_clust as circ

    circ.prep_for_circus()
    circ.start_the_show()


class Dataset(objects.data_object):
    """
    Parameters
    ----------
    root_dir : str (optional)
        absolute path to a recording directory, if left empty a filechooser will pop up.
    """

    PROCESSING_STEPS = [
        "initialize_parameters",
        "extract_data",
        "create_trial_list",
        "mark_dead_channels",
        "spike_detection",
        "spike_clustering",
        "cleanup_clustering",
        "spike_sorting",
        "make_unit_plots",
        "units_similarity",
        "make_unit_arrays",
        "make_psth_arrays",
        "plot_psths",
        "palatability_calculate",
        "palatability_plot",
        "overlay_psth",
    ]

    def __init__(
        self,
        root_dir=None,
        data_name=None,
        shell=False,
        file_type=".smr",
    ):
        """
        Initialize dataset object from file_dir, grabs basename from name of
        directory and initializes basic analysis parameters

        Parameters
        ----------
        root_dir : str (optional), file directory for intan recording data

        """
        super().__init__(data_type="dataset", root_dir=root_dir, data_name=data_name, shell=shell)

        # TODO: add check for file type + additional file types

        h5_file = get_h5_filename(self.root_dir)
        if h5_file is None:
            h5_file = self.root_dir / f"{self.data_name}.h5"
            print(
                f"No .h5 file found in {self.root_dir}. \n"
                f"Creating new h5 file: {h5_file}"
            )
        else:
            print(f"Existing h5 file found. Using {h5_file}.")

        self.h5_file = h5_file

        self.rec_info = {"file_type": file_type}

        self.dataset_creation_date = dt.datetime.today()
        self.processing_steps = Dataset.PROCESSING_STEPS.copy()
        self._process_status = dict.fromkeys(self.processing_steps, False)

    @property
    def process_status(self):
        return self._process_status

    def _change_root(self, new_root=None):
        old_root = self.root_dir
        new_root = super()._change_root(new_root)
        self.h5_file = self.h5_file.replace(old_root, new_root)
        return new_root

    def initialize_parameters(
        self,
        data_quality="hp",
        emg_port=None,  #  TODO: add emg_port to rec_info
        emg_channels=None,  #  TODO: add emg_channels to rec_info
        dig_in_names=None,  # This should be events, any input from the user
        dig_out_names=None,
        shell=False,
        accept_params=False,
    ):
        """
        Initalizes basic default analysis parameters and allows customization
        of parameters

        Parameters (all optional)
        -------------------------
        data_quality : {'clean', 'noisy', 'hp'}
            keyword defining which default set of parameters to use to detect
            headstage disconnection during clustering
            default is 'clean'. Best practice is to run blech_clust as 'clean'
            and re-run as 'noisy' if too many early cutoffs occurr.
            Alternately run as 'hp' (high performance)
            default parameter sets found in spk_io.defualts.clustering_params.json
        emg_port : int
            port number of EMG data
            default is None
        emg_channels : list of int
            channel or channels of EMGs on port specified
            default is None
        shell : bool
            False (default) for GUI. True for command-line interface
        dig_in_names : list of str
            Names of digital inputs. Must match number of digital inputs used
            in recording.
            None (default) queries user to name each dig_in
        dig_out_names : list of str
            Names of digital outputs. Must match number of digital outputs in
            recording.
            None (default) queries user to name each dig_out
        accept_params : bool
            True automatically accepts default parameters where possible,
            decreasing user queries
            False (default) will query user to confirm or edit parameters for
            clustering, spike array and psth creation and palatability/identity
            calculations
        """

        # intan files are stored as in .dat format readable from a text parser,
        # but spike2 datafiles require the SonPy library to extract.

        file_dir = Path(self.root_dir)
        data_files = [f for f in file_dir.iterdir() if f.suffix == ".smr"]

        if len(data_files) > 1:
            print(
                f"Multiple Spike2 files found in {file_dir}. \n"
                f"Select the one you want to load."
            )
            file = userio.select_from_list(
                "Select Spike2 file", data_files, "Spike2 File Selection", shell=shell
            )
        else:
            file = data_files[0]

        print("Extracting information from Spike2 file")
        self.data = Spike2Data(filepath=file,)
        self.electrode_mapping = self.data.load_mapping()  # this gets metadata without loading data into memory
        self.electrode_mapping = self.electrode_mapping[self.electrode_mapping['unit'] == True] ## separate out units from other channels

        print(self.electrode_mapping)

        # add the data dataframe to the rec_info dictionary
        self.rec_info.update(self.electrode_mapping.to_dict())
        self.rec_info.update(
            {
                "rec_length": self.data.rec_length,
                "num_electrodes": len(self.electrode_mapping),
                "idx_electrodes": self.electrode_mapping['electrode'].values,
                "name_electrodes": self.electrode_mapping['name'].values,
            }
        )
        self.rec_df = pd.DataFrame.from_dict(self.rec_info, orient='index').T

        # Get default parameters from files
        clustering_params = load_params(
            "clustering_params", file_dir, default_keyword=data_quality
        )
        spike_array_params = load_params("spike_array_params", file_dir)
        psth_params = load_params("psth_params", file_dir)
        pal_id_params = load_params("pal_id_params", file_dir)
        unit_sampling_rates = self.electrode_mapping["sampling_rate"].unique()
        if len(unit_sampling_rates) > 1:
            raise ValueError(
                "Multiple sampling rates found in units. " "This is not yet supported."
            )

        spike_array_params["sampling_rate"] = unit_sampling_rates[0]
        clustering_params["sampling_rate"] = spike_array_params["sampling_rate"]
        clustering_params["file_dir"] = file_dir

        self.spike_array_params = spike_array_params

        # Confirm parameters
        if not accept_params:
            conf = userio.confirm_parameter_dict
            clustering_params = conf(
                clustering_params, "Clustering Parameters", shell=shell
            )
            self.edit_spike_array_params(shell=shell)
            psth_params = conf(psth_params, "PSTH Parameters", shell=shell)
            pal_id_params = conf(
                pal_id_params,
                "Palatability/Identity Parameters\n"
                "Valid unit_type is Single, Multi or All",
                shell=shell,
            )

        # Store parameters
        self.clustering_params = clustering_params
        self.pal_id_params = pal_id_params
        self.psth_params = psth_params
        self._write_all_params_to_json()
        self.process_status["initialize_parameters"] = True
        self.save()

    def _setup_digital_mapping(self, dig_type, dig_in_names=None, shell=False):
        """sets up dig_in_mapping dataframe  and queries user to fill in columns

        Parameters
        ----------
        dig_in_names : list of str (optional)
        shell : bool (optional)
            True for command-line interface
            False (default) for GUI
        """
        df = pd.DataFrame()
        df["channel"] = self.rec_info.get("dig_%s" % dig_type)
        # Names
        if dig_in_names:
            df["name"] = dig_in_names
        else:
            df["name"] = ""

        df["exclude"] = False
        # Re-format for query
        idx = df.index
        df.index = ["dig_%s_%i" % (dig_type, x) for x in df.channel]
        dig_str = dig_type + "put"

        # Query for user input
        usrprompt = (
            f"Digital {dig_str} Parameters\nSet palatability ranks from 1 to {len(df)}"
        )
        tmp = userio.fill_dict(df.to_dict(), prompt=usrprompt, shell=shell)

        # Reformat for storage
        df2 = pd.DataFrame.from_dict(tmp)
        df2 = df2.sort_values(by=["channel"])
        df2.index = idx
        if dig_type == "in":
            df2["palatability_rank"] = df2["palatability_rank"].fillna(-1).astype("int")

        if dig_type == "in":
            self.dig_in_mapping = dim = df2.copy()
            self.spike_array_params["laser_channels"] = dim.channel[
                dim["laser"]
            ].to_list()
            self.spike_array_params["dig_ins_to_use"] = dim.channel[
                dim["spike_array"]
            ].to_list()
            write_params_to_json(
                "spike_array_params", self.root_dir, self.spike_array_params
            )
        else:
            self.dig_out_mapping = df2.copy()

        if os.path.isfile(self.h5_file):
            write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, dig_type)

    def edit_spike_array_params(self, shell=False):
        """Edit spike array parameters and adjust dig_in_mapping accordingly

        Parameters
        ----------
        shell : bool, whether to use CLI or GUI
        """
        if not hasattr(self, "dig_in_mapping"):
            self.spike_array_params = None
            return

        sa = deepcopy(self.spike_array_params)
        tmp = userio.fill_dict(sa, "Spike Array Parameters\n(Times in ms)", shell=shell)
        if tmp is None:
            return

        dim = self.dig_in_mapping
        dim["spike_array"] = False
        if tmp["dig_ins_to_use"] != [""]:
            tmp["dig_ins_to_use"] = [int(x) for x in tmp["dig_ins_to_use"]]
            dim.loc[
                [x in tmp["dig_ins_to_use"] for x in dim.channel], "spike_array"
            ] = True

        dim["laser"] = False
        if tmp["laser_channels"] != [""]:
            tmp["laser_channels"] = [int(x) for x in tmp["laser_channels"]]
            dim.loc[[x in tmp["laser_channels"] for x in dim.channel], "laser"] = True

        self.spike_array_params = tmp.copy()
        write_params_to_json("spike_array_params", self.root_dir, tmp)
        if os.path.isfile(self.h5_file):
            write_digital_map_to_h5(self.h5_file, self.dig_in_mapping, "in")

        self.save()

    def edit_clustering_params(self, shell=False):
        """Allows user interface for editing clustering parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        """
        tmp = userio.fill_dict(
            self.clustering_params, "Clustering Parameters\n(Times in ms)", shell=shell
        )
        if tmp:
            self.clustering_params = tmp
            write_params_to_json("clustering_params", self.root_dir, tmp)

        self.save()

    def edit_psth_params(self, shell=False):
        """Allows user interface for editing psth parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        """
        tmp = userio.fill_dict(
            self.psth_params, "PSTH Parameters\n(Times in ms)", shell=shell
        )
        if tmp:
            self.psth_params = tmp
            write_params_to_json("psth_params", self.root_dir, tmp)

        self.save()

    def edit_pal_id_params(self, shell=False):
        """Allows user interface for editing palatability/identity parameters

        Parameters
        ----------
        shell : bool (optional)
            True if you want command-line interface, False for GUI (default)
        """
        tmp = userio.fill_dict(
            self.pal_id_params,
            "Palatability/Identity Parameters\n(Times in ms)",
            shell=shell,
        )
        if tmp:
            self.pal_id_params = tmp
            write_params_to_json("pal_id_params", self.root_dir, tmp)

        self.save()

    def __str__(self):
        """
        Put all information about dataset in string format

        Returns
        -------
        str : representation of dataset object
        """
        out1 = super().__str__()
        out = [out1]
        out.append(
            "\nObject creation date: " + self.dataset_creation_date.strftime("%m/%d/%y")
        )

        if hasattr(self, "raw_h5_file"):
            out.append("Deleted Raw h5 file: " + self.raw_h5_file)

        out.append("h5 File: " + str(self.h5_file))
        out.append("")

        out.append("--------------------")
        out.append("Processing Status")
        out.append("--------------------")
        out.append(pt.print_dict(self.process_status))
        out.append("")

        if not hasattr(self, "rec_info"):
            return "\n".join(out)

        info = self.rec_info

        # if self.emg_mapping:
        #     out.append("--------------------")
        #     out.append("EMG")
        #     out.append("--------------------")
        #     out.append(pt.print_dataframe(self.emg_mapping))
        #     out.append("")

        if info.get("dig_in"):
            out.append("--------------------")
            out.append("Digital Input")
            out.append("--------------------")
            out.append(pt.print_dataframe(self.dig_in_mapping))
            out.append("")

        if info.get("dig_out"):
            out.append("--------------------")
            out.append("Digital Output")
            out.append("--------------------")
            out.append(pt.print_dataframe(self.dig_out_mapping))
            out.append("")

        out.append("--------------------")
        out.append("Clustering Parameters")
        out.append("--------------------")
        try:
            out.append(pt.print_dict(self.clustering_params))
        except AttributeError:
            out.append("Clustering parameters not yet set")
        out.append("")

        out.append("--------------------")
        out.append("Spike Array Parameters")
        out.append("--------------------")
        try:
            out.append(pt.print_dict(self.spike_array_params))
        except AttributeError:
            out.append("Spike array parameters not yet set")
        out.append("")

        try:
            out.append("--------------------")
            out.append("PSTH Parameters")
            out.append("--------------------")
            out.append(pt.print_dict(self.psth_params))
            out.append("")
        except AttributeError:
            pass

        try:
            out.append("--------------------")
            out.append("Palatability/Identity Parameters")
            out.append("--------------------")
            out.append(pt.print_dict(self.pal_id_params))
            out.append("")
        except AttributeError:
            pass

        return "\n".join(out)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def _write_all_params_to_json(self):
        """
        Writes all parameters to json files in analysis_params folder in the
        recording directory
        """
        print("Writing all parameters to json file in analysis_params folder...")
        clustering_params = self.clustering_params
        spike_array_params = self.spike_array_params if hasattr(self, "spike_array_params") else {}
        psth_params = self.psth_params
        pal_id_params = self.pal_id_params
        rec_dir = self.root_dir
        write_params_to_json("clustering_params", rec_dir, clustering_params)
        write_params_to_json("spike_array_params", rec_dir, spike_array_params)
        write_params_to_json("psth_params", rec_dir, psth_params)
        write_params_to_json("pal_id_params", rec_dir, pal_id_params)

    def extract_data(self, filename=None,):
        """
        Create a new H5 file with a pre-defined structure:

        - /raw
            - /electrode0[EArray]
            - /electrode1[EArray]
            - ...
        - /time
            - /time_vector[EArray]



        .. seealso::
            :func:`cpl_extract.spk_io.h5io.create_empty_data_h5`
            :func:`cpl_extract.spk_io.h5io.create_hdf_arrays`

        Create hdf5 store for data and read in data files and create
        subfolders for processing outputs.

        Parameters
        ----------
        filename : str (optional)
            name of h5 file to create. If none is specified, the default is
            the name of the recording directory with a .h5 extension
        """

        file = filename if filename else self.h5_file
        tmp = spk_io.create_empty_data_h5(
            filename=file,
        )
        if tmp is None:
            print("Data already extracted. Skipping.")
            return

        spk_io.create_hdf_arrays(
            file_name=file,
            rec_info=self.rec_info,
            electrode_mapping=self.electrode_mapping,
        )

        print("Extracting data from Spike2 file")
        self._process_spike2data()
        self.process_status["extract_data"] = True
        self.save()

        print("\nData Extraction Complete\n--------------------")

    def _process_spike2data(self):
        """
        Extract all data from Spike2 file and save to h5 file.

        Called only when extracting data from Spike2 file and extract_data() is called"""

        electrodes = self.electrode_mapping["electrode"].unique()
        _time_flag = False

        for electrode_idx in electrodes:
            ic(f"Extracting electrode {electrode_idx}/{electrodes[-1]}")

            this_electrode = self.electrode_mapping[self.electrode_mapping["electrode"] == electrode_idx]
            electrode = this_electrode["electrode"].iloc[0]
            unit_fs = this_electrode["sampling_rate"].iloc[0]
            data = [chunk for chunk in self.data.read_data_in_chunks(electrode_idx)]
            data = list(itertools.chain(*data))

            write_spike2_array_to_h5(self.h5_file, electrode, waves=data, fs=unit_fs)
            write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)
            if _time_flag is False:
                saved = write_time_vector_to_h5(self.h5_file, electrode, unit_fs)
                if saved:
                    ic("Time vector saved to h5 file")
                    _time_flag = True

    def create_trial_list(self):
        """
        Create lists of trials based on digital inputs and outputs and store to hdf5 store.

        Can only be run *after* data extraction.

        """
        if self.rec_info.get("dig_in"):
            in_list = create_trial_data_table(
                self.h5_file, self.dig_in_mapping, self.sampling_rate, "in"
            )
            self.dig_in_trials = in_list
        else:
            print("No digital input data found")

        if self.rec_info.get("dig_out"):
            out_list = create_trial_data_table(
                self.h5_file, self.dig_out_mapping, self.sampling_rate, "out"
            )
            self.dig_out_trials = out_list
        else:
            print("No digital output data found")


        self.process_status["create_trial_list"] = True
        self.save()

    def mark_dead_channels(self, dead_channels=None, shell=False):
        """
        Plots small piece of raw traces and a metric to help identify dead
        channels. Once user marks channels as dead a new column is added to
        electrode mapping

        Parameters
        ----------
        dead_channels : list of int, optional
            if this is specified then nothing is plotted, those channels are
            simply marked as dead
        shell : bool, optional
        """
        print("Marking dead channels\n----------")
        # em = self.electrode_mapping.copy()
        em = self.electrode_mapping.copy()
        if dead_channels is None:
            userio.tell_user("Making traces figure for dead channel detection...", shell=True)

            save_file = os.path.join(self.root_dir, "electrode_Traces.png")
            fig, ax = plot_traces_and_outliers(self.h5_file, save_file=save_file)
            if not shell:
                # Better to open figure outside of python since it's a lot of
                # data on figure and matplotlib is slow
                # xd-open is a linux command to open file with default program, change for other OS
                subprocess.call(["xdg-open", save_file])
            else:
                userio.tell_user(
                    "Saved figure of traces to %s for reference" % save_file,
                    shell=shell,
                )

            choice = userio.select_from_list(
                "Select dead channels:",
                em.electrode.to_list(),
                "Dead Channel Selection",
                multi_select=True,
                shell=shell,
            )
            dead_channels = list(map(int, choice)) if choice else []

        print(f"Marking the following units/electrodes as dead: \n" f"{dead_channels}")
        em["dead"] = False
        em.loc[dead_channels, "dead"] = True
        self.electrode_mapping = em
        if os.path.isfile(self.h5_file):
            write_electrode_map_to_h5(self.h5_file, self.electrode_mapping)
        self.process_status["mark_dead_channels"] = True
        self.save()

        return dead_channels

    def detect_spikes(self, data_quality=None, multi_process=False, n_cores=None):
        """
        Run spike detection on each electrode.
        Works for both single recording clustering or multi-recording clustering.

        Parameters
        ----------
        data_quality : {'clean', 'noisy', None (default)}
            set if you want to change the data quality parameters for cutoff
            and spike detection before running clustering. These parameters are
            automatically set as "clean" during initial parameter setup
        multi_process : bool, False (default)
            set to True to run spike detection on multiple cores
        n_cores : int (optional)
            number of cores to use for parallel processing. default is max-1.
            has no effect if multi_process is False
        """
        if data_quality:
            tmp = load_params(
                "clustering_params", self.root_dir, default_keyword=data_quality
            )
            if tmp:
                self.clustering_params = tmp
                write_params_to_json("clustering_params", self.root_dir, tmp)
            else:
                raise ValueError(
                    "%s is not a valid data_quality preset. Must "
                    'be "clean" or "noisy" or None.'
                )

        print("\nRunning Spike Detection\n-------------------")
        print("Parameters\n%s" % pt.print_dict(self.clustering_params))

        # Create folders for saving things within recording dir
        data_dir = self.root_dir

        em = self.electrode_mapping
        if "dead" in em.columns:
            electrodes = em.electrode[em["dead"] == False].tolist()
        else:
            electrodes = em.electrode.tolist()

        if multi_process:
            spike_detectors = [
                clust.SpikeDetection(data_dir, x, self.clustering_params) for x in electrodes
            ]

            if n_cores is None or n_cores > cpu_count():
                n_cores = cpu_count() - 1

            results = Parallel(n_jobs=n_cores)(
                delayed(run_joblib_process)(sd) for sd in spike_detectors
            )
        else:
            results = [(None, None, None)] * (max(electrodes) + 1)
            spike_detectors = [
                clust.SpikeDetection(data_dir, x, self.clustering_params) for x in electrodes
            ]
            for sd in tqdm(spike_detectors):
                res = sd.run()
                results[res[0]] = res

        print("electrode    Result    Cutoff (s)")
        cutoffs = {}
        clust_res = {}
        clustered = []
        for x, y, z in results:
            if x is None:
                continue

            clustered.append(x)
            print("  {:<13}{:<10}{}".format(x, y, z))
            cutoffs[x] = z
            clust_res[x] = y

        print("1 - Success\n0 - No data or no spikes\n-1 - Error")

        em = self.electrode_mapping.copy()
        em["cutoff_time"] = em["electrode"].map(cutoffs)
        em["clustering_result"] = em["electrode"].map(clust_res)
        self.electrode_mapping = em.copy()
        self.process_status["spike_detection"] = True
        write_electrode_map_to_h5(self.h5_file, em)
        self.save()
        print("Spike Detection Complete\n------------------")
        return results

    def cluster_spikes(self, data_quality=None, multi_process=False, n_cores=None, umap=True, accept_params=False):
        """
        Write clustering parameters to file and
        Run process on each electrode using GNU parallel

        Parameters
        ----------
        data_quality : {'clean', 'noisy', None (default)}
            set if you want to change the data quality parameters for cutoff
            and spike detection before running clustering. These parameters are
            automatically set as "clean" during initial parameter setup
        accept_params : bool, False (default)
            set to True in order to skip popup confirmation of parameters when
            running
        """
        if not self.process_status["spike_detection"]:
            raise FileNotFoundError("Must run spike detection before clustering.")

        if data_quality:
            tmp = load_params(
                "clustering_params", self.root_dir, default_keyword=data_quality
            )
            if tmp:
                self.clustering_params = tmp
                write_params_to_json("clustering_params", self.root_dir, tmp)
            else:
                raise ValueError(
                    "%s is not a valid data_quality preset. Must "
                    'be "clean" or "noisy" or None.'
                )

        print("\nRunning Spike Clustering\n-------------------")
        print("Parameters\n%s" % pt.print_dict(self.clustering_params))

        # Get electrodes, throw out 'dead' electrodes
        em = self.electrode_mapping
        if "dead" in em.columns:
            electrodes = em.electrode[em["dead"] == False].tolist()
        else:
            electrodes = em.electrode.tolist()

        if not umap:
            clust_objs = [
                clust.CplClust(self.root_dir, x, params=self.clustering_params)
                for x in electrodes
            ]
        else:
            clust_objs = [
                clust.CplClust(
                    self.root_dir,
                    x,
                    params=self.clustering_params,
                    data_transform=clust.UMAP_METRICS,
                    n_pc=5,
                )
                for x in electrodes
            ]

        if multi_process:
            if n_cores is None or n_cores > cpu_count():
                n_cores = cpu_count() - 1

            results = Parallel(n_jobs=n_cores, verbose=10)(
                delayed(run_joblib_process)(co) for co in clust_objs
            )

        else:
            results = []
            for x in clust_objs:
                res = x.run()
                results.append(res)

        self.process_status["spike_clustering"] = True
        self.process_status["cleanup_clustering"] = False
        write_electrode_map_to_h5(self.h5_file, em)
        self.save()
        print("Clustering Complete\n------------------")

    def cleanup_clustering(self):
        """
        Consolidates memory monitor files, removes raw and referenced data
        and sets up the h5 data store for spike-sorting.
        """
        if self.process_status["cleanup_clustering"]:
            return

        h5_file = cleanup_clustering(self.root_dir, h5_file=self.h5_file)
        self.h5_file = h5_file
        self.process_status["cleanup_clustering"] = True
        self.save()

    def sort_spikes(self, electrode=None, all_electrodes=(), shell=False) -> (Tk, SpikeSorterGUI):
        if electrode is None:
            ic("No electrode specified. Asking user for elextrode.")
            if all_electrodes:
                ic(f"all_electrodes: {all_electrodes}")
                electrode_formatted_str = ", ".join([str(x) for x in all_electrodes])
            else:
                electrode_formatted_str = ", ".join([str(x) for x in self.electrode_mapping["electrode"].unique()])

            electrode = userio.get_user_input(f"Choose an electrode to process. \n"
                                              f"Possible electrodes: \n"
                                              f"{electrode_formatted_str}", shell=shell)
            if electrode is None or not electrode.isnumeric():
                return
            electrode = int(electrode)

        if not self.process_status["spike_clustering"]:
            raise ValueError("Must run spike clustering first.")

        if not self.process_status["cleanup_clustering"]:
            self.cleanup_clustering()

        sorter = clust.SpikeSorter(rec_dirs=self.root_dir, electrode=electrode, shell=shell)
        if not shell:
            root, sorting_GUI = launch_sorter_GUI(sorter)
            if root:
                root.mainloop()
        self.process_status["spike_sorting"] = True

    def units_similarity(self, similarity_cutoff=50, shell=False):
        if "SSH_CONNECTION" in os.environ:
            shell = True

        metrics_dir = os.path.join(self.root_dir, "sorted_unit_metrics")
        if not os.path.isdir(metrics_dir):
            raise ValueError(
                "No sorted unit metrics found. Must sort units before calculating similarity"
            )

        violation_file = os.path.join(metrics_dir, "units_similarity_violations.txt")
        violations, sim = calc_units_similarity(
            self.h5_file, self.sampling_rate, similarity_cutoff, violation_file
        )
        if len(violations) == 0:
            userio.tell_user("No similarity violations found!", shell=shell)
            self.process_status["units_similarity"] = True
            return violations, sim

        out_str = ["Units Similarity Violations Found:"]
        out_str.append("Unit_1    Unit_2    Similarity")
        for x, y in violations:
            u1 = h5io.parse_unit_number(x)
            u2 = h5io.parse_unit_number(y)
            out_str.append("   {:<10}{:<10}{}\n".format(x, y, sim[u1][u2]))

        out_str.append("Delete units with dataset.delete_unit(N)")
        out_str = "\n".join(out_str)
        userio.tell_user(out_str, shell=shell)
        self.process_status["units_similarity"] = True
        self.save()
        return violations, sim

    def delete_unit(self, unit_num, confirm=False, shell=False):
        if isinstance(unit_num, str):
            unit_num = h5io.parse_unit_number(unit_num)

        if unit_num is None:
            print("No unit deleted")
            return

        if not confirm:
            q = userio.ask_user(
                "Are you sure you want to delete unit%03i?" % unit_num,
                choices=["No", "Yes"],
                shell=shell,
            )
        else:
            q = 1

        if q == 0:
            print("No unit deleted")
            return
        else:
            tmp = h5io.delete_unit(self.root_dir, unit_num, h5_file=self.h5_file)
            if tmp is False:
                userio.tell_user(
                    "Unit %i not found in dataset. No unit deleted" % unit_num,
                    shell=shell,
                )
            else:
                userio.tell_user("Unit %i sucessfully deleted." % unit_num, shell=shell)

        self.save()

    def make_unit_arrays(self):
        """Make spike arrays for each unit and store in hdf5 store"""
        params = self.spike_array_params

        print("Generating unit arrays with parameters:\n----------")
        print(pt.print_dict(params, tabs=1))
        make_spike_arrays(self.h5_file, params)
        self.process_status["make_unit_arrays"] = True
        self.save()

    def make_unit_plots(self):
        """Make waveform plots for each sorted unit"""
        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, "unit_waveforms_plots")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)
        for i, row in unit_table.iterrows():
            make_unit_plots(self.root_dir, row["unit_name"], save_dir=save_dir)

        self.process_status["make_unit_plots"] = True
        self.save()

    def make_psth_arrays(self):
        """Make smoothed firing rate traces for each unit/trial and store in
        hdf5 store
        """
        params = self.psth_params
        dig_ins = self.dig_in_mapping.query("spike_array == True")
        for idx, row in dig_ins.iterrows():
            spike_analysis.make_psths_for_tastant(
                self.h5_file,
                params["window_size"],
                params["window_step"],
                row["channel"],
            )

        self.process_status["make_psth_arrays"] = True
        self.save()

    def make_rate_arrays(self):
        """
        Make firing rate arrays for each unit and store in hdf5 store
        """
        params = self.psth_params
        dig_ins = self.dig_in_mapping.query("spike_array == True")
        for idx, row in dig_ins.iterrows():
            dig_in_ch = row["channel"]
            print(dig_in_ch)
            make_rate_arrays(self.h5_file, dig_in_ch)

    def make_psth_plots(self, sd=True, save_prefix=None):
        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, "unit_psth_plots")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        dinmap = self.dig_in_mapping.query("spike_array ==True")

        for i, row in unit_table.iterrows():
            un = row.unit_num
            if save_prefix is None:
                save_file = os.path.join(save_dir, "unit_" + str(un) + "_PSTH.svg")
            else:
                save_file = os.path.join(
                    save_dir, save_prefix + "unit_" + str(un) + "_PSTH.svg"
                )
            plot_overlay_psth(
                rec_dir=self.root_dir,
                unit=un,
                plot_window=[-1500, 5000],
                bin_size=500,
                sd=sd,
                din_map=dinmap,
                save_file=save_file,
            )

    def make_raster_plots(self):
        """make raster plots with electrode noise for each unit"""

        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, "unit_raster_plots")

        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        for i, row in unit_table.iterrows():
            spike_times, _, _ = h5io.get_unit_spike_times(
                self.root_dir, row["unit_name"], h5_file=self.h5_file
            )

            waveforms, _, _ = h5io.get_unit_waveforms(
                self.root_dir, row["unit_name"], h5_file=self.h5_file
            )
            save_file = os.path.join(save_dir, row["unit_name"] + "_raster")
            plot_spike_raster([spike_times], [waveforms], save_file=save_file)

        self.save()

    def make_trial_raster_plots(self):
        """
        make raster plots for each neuron across dig_in_trials
        """
        unit_table = self.get_unit_table()
        save_dir = os.path.join(self.root_dir, "trial_raster_plots")

    def make_ensemble_raster_plots(self):
        save_dir = os.path.join(self.root_dir, "raster_plots")
        name = self.data_name
        save_file = os.path.join(save_dir, name + "_ensemble_raster")
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        plot_ensemble_raster(self, save_file)

    def palatability_calculate(self, shell=False):
        from cpl_extract.analysis import palatability_analysis as pal_analysis

        pal_analysis.palatability_identity_calculations(
            self.root_dir, params=self.pal_id_params
        )
        self.process_status["palatability_calculate"] = True
        self.save()

    def palatability_plot(self, shell=False):
        raise NotImplementedError
        # self.process_status["palatability_plot"] = True
        # self.save()

    def cleanup_lowSpiking_units(self, min_spikes=100):
        unit_table = self.get_unit_table()
        remove = []
        spike_count = []
        for unit in unit_table["unit_num"]:
            waves, descrip, fs = h5io.get_unit_waveforms(
                self.root_dir, unit, h5_file=self.h5_file
            )
            if waves.shape[0] < min_spikes:
                spike_count.append(waves.shape[0])
                remove.append(unit)

        for unit, count in zip(reversed(remove), reversed(spike_count)):
            print("Removing unit %i. Only %i spikes." % (unit, count))
            userio.tell_user(
                "Removing unit %i. Only %i spikes." % (unit, count), shell=True
            )
            self.delete_unit(unit, confirm=True, shell=True)

        userio.tell_user(
            "Removed %i units for having less than %i spikes."
            % (len(remove), min_spikes),
            shell=True,
        )

    def get_unit_table(self):
        """Returns a pandas dataframe with sorted unit information

        Returns
        --------
        pandas.DataFrame with columns:
            unit_name, unit_num, electrode, single_unit,
            regular_spiking, fast_spiking
        """
        unit_table = h5io.get_unit_table(self.root_dir, h5_file=self.h5_file)
        return unit_table

    def edit_unit_descriptor(self, unit_num, descriptor_key, descriptor_val):
        """
        use this to edit unit table, i.e. if you made a mistake labeling a neuron in spike sorting
        unit_num takes integers, corresponds to unit_num in get_unit_table()
        descriptor_key takes string, can be "single_unit", "regular_spiking", or "fast_spiking"
        descriptor_val takes boolean, can be True or False
        """
        h5io.edit_unit_descriptor(
            self.root_dir, unit_num, descriptor_key, descriptor_val, self.h5_file
        )
        print("descriptor edit success")

    def pre_process_for_clustering(self, shell=False, dead_channels=None):
        status = self.process_status
        if not status["initialize_parameters"]:
            self.initialize_parameters(shell=shell)

        if not status["extract_data"]:
            self.extract_data()

        if not status["create_trial_list"]:
            self.create_trial_list()

        if not status["mark_dead_channels"] and dead_channels != False:
            self.mark_dead_channels(dead_channels=dead_channels, shell=shell)

        try:  # not using at the moment
            if not status["common_average_reference"]:
                self.common_average_reference()
        except AttributeError:
            pass

        if not status["spike_detection"]:
            self.detect_spikes()

    def extract_and_circus_cluster(self, dead_channels=None, shell=True):
        print("Extracting Data...")
        self.extract_data()
        print("Marking dead channels...")
        self.mark_dead_channels(dead_channels, shell=shell)
        # TODO:
        # print("Common average referencing...")
        # self.common_average_reference()
        print("Initiating circus clustering...")
        circus = circus_clust(
            self.root_dir, self.data_name, self.sampling_rate, self.electrode_mapping
        )
        print("Preparing for circus...")
        circus.prep_for_circus()
        print("Starting circus clustering...")
        circus.start_the_show()
        print("Plotting cluster waveforms...")
        circus.plot_cluster_waveforms()

    def post_sorting(self):
        self.make_unit_plots()
        self.make_unit_arrays()
        self.units_similarity(shell=True)
        self.make_psth_arrays()
        self.make_raster_plots()

    def export_TrialSpikeArrays2Mat(self):
        h5 = tables.open_file(self.h5_file, "r+")
        taste_dig_in = h5.list_nodes("/spike_trains")
        loops = len(taste_dig_in)
        tbl = {}

        names = self.dig_in_mapping[self.dig_in_mapping["spike_array"] == True]["name"]
        key = names.values.tolist()
        spike_arrs = np.zeros(loops, dtype=np.object)

        for i in range(loops):
            if i < len(taste_dig_in):
                spike_arrs[i] = taste_dig_in[i].spike_array[:]

        nameparts = str.split(self.tbla_name, "_")
        tbl["ID"] = nameparts[0]
        tbl["tble"] = nameparts[-2]
        tbl["spikes"] = spike_arrs
        tbl["states"] = key

        ff = os.path.join(
            self.root_dir, "matlab_exports"
        )  # make variable with folder name (file folder)

        if not os.path.exists(ff):  # check if file folder exists, make if if not
            os.makedirs(ff)
            print("Directory created successfully")

        fn = self.tbla_name + ".mat"  # make file name
        fp = os.path.join(ff, fn)  # make file path
        sio.savemat(
            fp, {"tbla": tbl}
        )  # save tbla [tbl] with label "tbla", at file path fp
        print("spike trains successfully exported to " + fp)

        h5.flush()
        h5.close()

    def get_spike_array_table(self):
        h5 = tables.open_file(self.h5_file, "r+")
        taste_dig_in = h5.list_nodes("/spike_trains")
        loops = len(taste_dig_in)
        tbl = {}

        names = self.dig_in_mapping[self.dig_in_mapping["spike_array"] == True][
            "name"
        ].copy()
        key = names.values.tolist()
        spike_arrs = np.zeros(loops, dtype=np.object)

        for i in range(loops):
            if i < len(taste_dig_in):
                spike_arrs[i] = taste_dig_in[i].spike_array[:]

        h5.flush()
        h5.close()

        tbl["spikes"] = spike_arrs
        tbl["name"] = key

        tbl = pd.DataFrame.from_dict(tbl)
        tbl = tbl.explode("spikes")
        tbl["din_trial"] = tbl.groupby(["name"]).cumcount()

        din_trls = self.dig_in_trials.copy()
        din_trls["din_trial"] = din_trls.groupby(["name"]).cumcount()

        tbl = tbl.merge(din_trls, how="left", on=["din_trial", "name"])

        nameparts = str.split(self.data_name, "_")
        tbl["ID"] = nameparts[0]
        tbl["date"] = nameparts[-2]
        tbl["rec_dir"] = self.root_dir

        unt_tbl = self.get_unit_table().copy()
        elec_tbl = self.electrode_mapping.copy()
        elec_tbl = elec_tbl.rename(columns={"electrode": "electrode"})
        elec_tbl = elec_tbl[["electrode", "area"]]
        unt_tbl = unt_tbl.merge(elec_tbl, how="left", on="electrode")

        unt_tbl = unt_tbl.apply(lambda x: [x.tolist()], axis=0)
        unt_tbl = pd.concat([unt_tbl] * len(tbl), ignore_index=True)

        cols = ["unit_num", "electrode", "regular_spiking", "fast_spiking", "area"]
        tbl[cols] = unt_tbl[cols]

        cols.append("spikes")
        tbl = tbl.explode(cols)

        return tbl

    def completed_steps(self):
        if not hasattr(self, "process_status"):
            return []
        return [k for k, v in self.process_status.items() if v in [True, "True", "true"]]

    def incomplete_steps(self):
        if not hasattr(self, "process_status"):
            return []
        return [k for k, v in self.process_status.items() if v in [False, "False", "false"]]


def run_joblib_process(process):
    res = process.run()
    return res

def port_in_dataset(rec_dir=None, shell=False):
    """Import an existing dataset into this framework"""
    if rec_dir is None:
        rec_dir = userio.get_filedirs("Select recording directory", shell=shell)
        if rec_dir is None:
            return None

    dat = Dataset(rec_dir, shell=shell)
    # Check files that will be overwritten: log_file, save_file
    if os.path.isfile(dat.save_file):
        prompt = (
            "%s already exists. Continuing will overwrite this. Continue?"
            % dat.save_file
        )
        q = userio.ask_user(prompt, shell=shell)
        if q == 0:
            print("Aborted")
            return None

    if os.path.isfile(dat.log_file):
        prompt = (
            "%s already exists. Continuing will append to this. Continue?"
            % dat.log_file
        )
        q = userio.ask_user(prompt, shell=shell)
        if q == 0:
            print("Aborted")
            return None

    with open(dat.log_file, "a") as f:
        print(
            "\n==========\nPorting dataset into cpl_extract format\n==========\n",
            file=f,
        )
        print(dat, file=f)

    # Check for info.rhd file or query needed info
    info_rhd = os.path.join(dat.root_dir, "info.rhd")
    if os.path.isfile(info_rhd):
        dat.initialize_parameters(shell=shell)
    else:
        raise FileNotFoundError(f"{info_rhd} is required for proper dataset creation")

    status = dat.process_status

    user_status = status.copy()
    user_status = userio.fill_dict(
        user_status,
        "Which processes have already been " "done to the data?",
        shell=shell,
    )

    status.update(user_status)
    # if h5 exists data must have been extracted

    if not os.path.isfile(dat.h5_file) or status["extract_data"] == False:
        dat.save()
        return dat

    # write eletrode map and digital input & output maps to hf5
    node_list = spk_io.h5io.get_node_list(dat.h5_file)

    if "electrode_map" not in node_list:
        spk_io.h5io.write_electrode_map_to_h5(dat.h5_file, dat.electrode_mapping)

    if dat.rec_info.get("dig_in") is not None and "digital_input_map" not in node_list:
        spk_io.h5io.write_digital_map_to_h5(dat.h5_file, dat.dig_in_mapping, "in")

    if (
        dat.rec_info.get("dig_out") is not None
        and "digital_output_map" not in node_list
    ):
        spk_io.h5io.write_digital_map_to_h5(dat.h5_file, dat.dig_out_mapping, "out")

    if ("trial_info" not in node_list) and ("digital_in" in node_list):
        dat.create_trial_list()
    else:
        status["create_trial_list"] == True

    dat.save()

    if status["spike_clustering"] and not status["spike_sorting"]:
        # Move files into correct structure to support spike sorting
        for i, row in dat.electrode_mapping.iterrows():
            el = row["electrode"]
            src = [
                os.path.join(dat.root_dir, "clustering_results", f"electrode{el}"),
                os.path.join(dat.root_dir, "Plots", f"{el}", "Plots"),
                os.path.join(
                    dat.root_dir, "Plots", f"{el}", "Plots", "cutoff_time.png"
                ),
                os.path.join(
                    dat.root_dir, "Plots", f"{el}", "Plots", "pca_variance.png"
                ),
                os.path.join(dat.root_dir, "spike_waveforms", f"electrode{el}"),
                os.path.join(dat.root_dir, "spike_times", f"electrode{el}", "spike_times.npy"),
            ]
            clust_dir = os.path.join(dat.root_dir, "BlechClust", f"electrode_{el}")
            detect_dir = os.path.join(dat.root_dir, "spike_detection", f"electrode_{el}")
            dest = [
                os.path.join(clust_dir, "clustering_results"),
                os.path.join(clust_dir, "plots"),
                os.path.join(detect_dir, "plots"),
                os.path.join(detect_dir, "plots"),
                os.path.join(detect_dir, "data"),
                os.path.join(detect_dir, "data"),
            ]
            for s, d in zip(src, dest):
                if not os.path.exists(s):
                    continue

                if not os.path.isdir(os.path.dirname(d)):
                    os.makedirs(os.path.dirname(d))

                shutil.copytree(s, d)

            # Make params files
            params = dat.clustering_params.copy()

            sd_fn = os.path.join(dat.root_dir, "analysis_params", "spike_detection_params.json")
            if not os.path.isfile(sd_fn):
                sd_params = {}
                sd_params["voltage_cutoff"] = params["data_params"][
                    "V_cutoff for disconnected headstage"
                ]
                sd_params["max_breach_rate"] = params["data_params"][
                    "Max rate of cutoff breach per second"
                ]
                sd_params["max_secs_above_cutoff"] = params["data_params"][
                    "Max allowed seconds with a breach"
                ]
                sd_params["max_mean_breach_rate_persec"] = params["data_params"][
                    "Max allowed breaches per second"
                ]
                band_lower = params["bandpass_params"]["Lower freq cutoff"]
                band_upper = params["bandpass_params"]["Upper freq cutoff"]
                sd_params["bandpass"] = [band_lower, band_upper]
                snapshot_pre = params["spike_snapshot"]["Time before spike (ms)"]
                snapshot_post = params["spike_snapshot"]["Time after spike (ms)"]
                sd_params["spike_snapshot"] = [snapshot_pre, snapshot_post]
                sd_params["sampling_rate"] = params["sampling_rate"]
                write_dict_to_json(sd_params, sd_fn)

            c_fn = os.path.join(clust_dir, "BlechClust_params.json")
            if not os.path.isfile(c_fn):
                c_params = params.copy()
                c_params["max_clusters"] = params["clustering_params"][
                    "Max Number of Clusters"
                ]
                c_params["max_iterations"] = params["clustering_params"][
                    "Max Number of Iterations"
                ]
                c_params["threshold"] = params["clustering_params"][
                    "Convergence Criterion"
                ]
                c_params["num_restarts"] = params["clustering_params"][
                    "GMM random restarts"
                ]
                c_params["wf_amplitude_sd_cutoff"] = params["data_params"][
                    "Intra-cluster waveform amp SD cutoff"
                ]
                write_dict_to_json(c_params, c_fn)

            # To make: clust_dir/clustering_results/ clustering_results.json, rec_key.json, spike_id.npy
            # To make: detect_dir/data/cutoff_time.txt and detection_threshold.txt
            sd = clust.SpikeDetection(dat.root_dir, el, overwrite=False)
            sd.run()  # should only filter referenced electrode trace and get cutoff and threshold
            bc = clust.CplClust(dat.root_dir, el)

    # Add array_time to spike_arrays/dig_in_#
    if "spike_trains" in node_list:
        digs = set(x.split(".")[1] for x in node_list if "spike_trains." in x)
        params = dat.spike_array_params
        array_time = np.arange(-params["pre_stimulus"], params["post_stimulus"], 1)
        for x in digs:
            if f"spike_trains.{x}.array_time" not in node_list:
                write_array_to_hdf5(dat.h5_file, f"/spike_trains/{x}", "array_time", array_time)

        for x in dat.processing_steps:
            status[x] = True
            if x == "make_unit_arrays":
                break

        dat.save()

    return dat
