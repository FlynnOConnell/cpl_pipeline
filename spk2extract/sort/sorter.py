# -*- coding: utf-8 -*-
"""
AutoSort: A Python package for automated spike sorting of extracellular recordings.
"""
from __future__ import annotations

# Standard Library Imports
import configparser
import itertools
import os
import shutil
import warnings
from datetime import date
from pathlib import Path
from typing import NamedTuple

# External Dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from sklearn.decomposition import PCA

from logger import logger
import cluster as clust  # avoid naming conflicts
from directory_manager import DirectoryManager
from plot import plot_cluster_features, plot_mahalanobis_to_cluster
from spk_config import SortConfig
from utils.progress import ProgressBarManager


# Factory
def sort(
    filename: str | Path,
    data: dict | NamedTuple,
    sampling_rate: float,
    params: SortConfig,
    dir_manager: DirectoryManager,
    chan_num: int,
    overwrite: bool = False,
):
    """
    Factory method for running the spike sorting process on a single channel.

    . .note::
    Data can be in several formats:
    1) Dictionary with keys 'spikes' and 'times'
    2) Named tuple with fields 'spikes' and 'times'

    # TODO: Add support for pandas DataFrame and other input methods

    Parameters
    ----------
    filename : str or Path
        Name of the file to be sorted.
    data : dict or NamedTuple
        Dictionary or namedtuple containing the data to be sorted.
    sampling_rate : float
        Sampling rate for this channel.
    params : SortConfig
        Configuration parameters for the spike sorting process.
    dir_manager : DirectoryManager
        DirectoryManager object for managing the output directories.
    chan_num : int
        Channel number to be sorted.
    overwrite : bool
        Whether to overwrite existing files. Default is False.
    """
    proc = ProcessChannel(filename, data, sampling_rate, params, dir_manager, chan_num, overwrite=overwrite)
    proc.process_channel()


def infofile(
    filename: str, path: str | Path, sort_time: float | str, params: SortConfig
):
    """
    Dumps run info to a .info file.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file being processed.
    path : str or Path
        The directory path where the .info file will be saved.
    sort_time : float or str
        Time taken for sorting.
    params : SortConfig
        Instance of SpkConfig with parameters used for sorting.

    Returns
    -------
    None
    """
    config = configparser.ConfigParser()
    config["METADATA"] = {
        "h5 File": filename,
        "Run Time": sort_time,
        "Run Date": date.today().strftime("%m/%d/%y"),
    }
    config["PARAMS USED"] = params.get_all()
    with open(
        path + "/" + os.path.splitext(filename)[0] + "_" + "sort.info", "w"
    ) as info_file:
        config.write(info_file)

# TODO: Making this a class doesn't accomplish much, refactor to functions
class ProcessChannel:
    """
    Class for running the spike sorting process on a single channel.

    Parameters
    ----------
    filename : str
        Name of the file to be processed.
    data : ndarray
        Raw data array.
    params : SortConfig
        Instance of SpkConfig holding processing parameters.
    dir_manager : DirectoryManager
        Directory manager object.
    chan_num : int
        Channel number to be processed.

    Attributes
    ----------
    filename : str
        Name of the file to be processed.
    spikes : ndarray
        Array of spike waveforms, from data['spikes'] or data.spikes.
    times : ndarray
        Array of spike times, from data['times'] or data.times.
    params : dict
        Dictionary of processing parameters.
    dir_manager : DirectoryManager
        Directory manager object.
    chan_num : int
        Channel number to be processed.
    sampling_rate : float
        Sampling rate (Default is 18518.52).

    Methods
    -------
    pvar()
        Returns the percent variance explained by the principal components.
    usepvar()
        Returns whether to use the percent variance explained by the principal components.
    userpc()
        Returns the number of principal components to use.
    max_clusters()
        Returns the total number of clusters to be sorted.
    max_iterations()
        Returns the maximum number of iterations to run the GMM.
    thresh()
        Returns the convergence criterion for the GMM.
    num_restarts()
        Returns the number of random restarts to run the GMM.
    wf_amplitude_sd_cutoff()
        Returns the number of standard deviations above the mean to reject waveforms.
    artifact_removal()
        Returns the number of standard deviations above the mean to reject waveforms.
    pre_time()
        Returns the number of standard deviations above the mean to reject waveforms.
    post_time()
        Returns the number of standard deviations above the mean to reject waveforms.
    bandpass()
        Returns the low and high cutoff frequencies for the bandpass filter.
    spike_detection()
        Returns the number of standard deviations above the mean to reject waveforms.
    std()
        Returns the number of standard deviations above the mean to reject waveforms.
    max_breach_rate()
        Returns the maximum breach rate.
    max_breach_count()
        Returns the maximum breach count.
    max_breach_avg()
        Returns the maximum breach average.
    voltage_cutoff()
        Returns the voltage cutoff.

    """

    def __init__(self, filename, data, sampling_rate, params, dir_manager, chan_num, overwrite=False):
        """
        Process a single channel.

        Parameters
        ----------
        filename : str
            Name of the file to be processed.
        data : dict | NamedTuple
            Raw data.
        sampling_rate : float
            Sampling rate for this channel.
        params : SortConfig
            SpkConfig instance of processing parameters.
        dir_manager : DirectoryManager
            Directory manager object.
        chan_num : int
            Channel number to be processed.
        overwrite : bool
            Whether to overwrite existing files.

        """
        self.data_columns = None
        self.metrics = None
        self.filename = filename
        if isinstance(data, dict):
            self.spikes = data["data"]
            self.times = data["times"]
        elif isinstance(data, NamedTuple):
            self.spikes = data[0]
            self.times = data[1]
        else:
            raise TypeError(
                "Data must be a dictionary/namedtuple with keys/fields 'spikes' and 'times'"
            )
        self.params = params
        self.sampling_rate = sampling_rate
        self.chan_num = chan_num
        self.dir_manager = dir_manager
        self.overwrite = overwrite
        self._data_dir =  Path(os.path.join(self.dir_manager.base_path, 'data', self.get_chan_str()))
        self._plots_dir = Path(os.path.join(self.dir_manager.base_path, 'plots', self.get_chan_str()))
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._files = {'raw_waveforms': os.path.join(self._data_dir, 'raw_waveforms.npy'),
                       'raw_times' : os.path.join(self._data_dir, 'raw_times.npy'),
                       'raw_metrics' : os.path.join(self._data_dir, 'metrics.npy'),
                       'raw_pca' : os.path.join(self._data_dir, 'raw_waveforms_pca.npy'),
                       # 'slopes' : os.path.join(self._data_dir, 'spike_slopes.npy'),
                       # 'recording_cutoff' : os.path.join(self._data_dir, 'cutoff_time.txt'),
                       # 'detection_threshold' : os.path.join(self._data_dir, 'detection_threshold.txt')
                     }

    def get_chan(self):
        """The channel number to use when saving."""
        return self.chan_num + 1

    def get_chan_str(self):
        """The channel number to use when saving."""
        return f"channel_{self.get_chan()}"

    @property
    def pvar(self):
        """
        Returns the percent of variance-explained at which to cut off the principal components.
        """
        return float(self.params.pca["variance-explained"])

    @property
    def usepvar(self):
        """
        Returns whether to use the percent variance explained by the principal components.
        """
        return int(self.params.pca["use-percent-variance"])

    @property
    def userpc(self):
        """
        Returns the number of principal components to use.
        """
        return int(self.params.pca["principal-component-n"])

    @property
    def min_clusters(self):
        """
        The minimum number of clusters to be sorted.
        """
        return int(self.params.cluster["min-clusters"])

    @property
    def max_clusters(self):
        """
        The maximum number of clusters to be sorted.
        """
        return int(self.params.cluster["max-clusters"])

    @property
    def max_iterations(self):
        """
        The maximum number of iterations to run the GMM.
        """
        return int(self.params.cluster["max-iterations"])

    @property
    def thresh(self):
        """
        The convergence criterion for the GMM.
        """
        return float(self.params.cluster["convergence-criterion"])

    @property
    def num_restarts(self):
        """
        The number of random restarts to run the GMM.
        """
        return int(self.params.cluster["random-restarts"])

    @property
    def wf_amplitude_sd_cutoff(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.cluster["intra-cluster-cutoff"])

    @property
    def artifact_removal(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.cluster["artifact-removal"])

    @property
    def pre_time(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.spike["pre_time"])

    @property
    def post_time(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.spike["post_time"])

    @property
    def bandpass(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.filter["low-cutoff"]), float(
            self.params.filter["high-cutoff"]
        )

    @property
    def spike_detection(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return int(self.params.detection["spike-detection"])

    @property
    def std(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return int(self.params.detection["spike-detection"])

    @property
    def max_breach_rate(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.breach["max-breach-rate"])

    @property
    def max_breach_count(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.breach["max-breach-count"])

    @property
    def max_breach_avg(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.breach["max-breach-avg"])

    @property
    def voltage_cutoff(self):
        """
        The number of standard deviations above the mean to reject waveforms.
        """
        return float(self.params.breach["voltage-cutoff"])

    def process_channel(
        self,
    ):
        """
        Executes spike sorting for a single recording channel.

        This method performs the entire spike sorting workflow for one channel.
        It includes dejittering, PCA transformation, and GMM-based clustering.
        If no spikes are found in the data, it writes out a warning and returns.

        Returns
        -------
        None
            Writes intermediate data and results to disk, does not return any value.

        Raises
        ------
        Warning
            Raises a warning if no spikes are found in the channel.

        Notes
        -----
        The function does the following main steps:

        1. Checks for spikes in the data. If none are found, writes a warning and returns.
        2. Dejitters spikes and extracts their amplitudes.
        3. Saves the spike waveforms and times as `.npy` files.
        4. Scales the spikes by waveform energy and applies PCA.
        5. Saves the PCA-transformed waveforms as `.npy` files.
        6. Performs GMM-based clustering on the PCA-transformed data.

        """
        while True:
            if not isinstance(self.spikes, np.ndarray):
                self.spikes = np.array(self.spikes)

            logger.info(f"|- --- Analyzing channel {self.chan_num + 1} --- -|")
            self.check_spikes(self.spikes, self.times)
            if not self.overwrite:
                if all([os.path.exists(f) for f in self._files.values()]):
                    logger.info(f"|- --- Channel {self.chan_num + 1} already processed, skipping --- -|")
                    break

            # PCA / UMAP
            filt = clust.filter_signal(
                self.spikes,
                self.sampling_rate,
                freq=(300.0, 3000.0)
            )

            spikes, times, threth  = clust.detect_spikes(
                filt,
                (0.5, 1.0),
                self.sampling_rate,
            )
            scaled_slices = clust.scale_waveforms(spikes)

            pca = PCA()
            pca_slices = pca.fit_transform(scaled_slices)
            pca_cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            pca_graph_vars = list(pca_cumulative_var[0 : np.where(pca_cumulative_var > 0.999)[0][0] + 1])
            pca_n_pc = (
                np.where(pca_cumulative_var > self.pvar)[0][0] + 1
                if self.usepvar == 1
                else self.userpc
            )

            pca_var_explained = float(pca_cumulative_var[pca_n_pc - 1])

            self.metrics, self.data_columns = clust.compute_waveform_metrics(
                scaled_slices, pca_n_pc, True
            )

            fig = plt.figure()
            x = np.arange(0, len(pca_graph_vars) + 1)
            pca_graph_vars.insert(0, 0)
            plt.plot(x, pca_graph_vars)
            plt.vlines(pca_n_pc, 0, 1, colors="r")
            plt.annotate(
                str(pca_n_pc)
                + " PC's used for GMM.\nVariance explained= "
                + str(round(pca_var_explained, 3))
                + "%.",
                (pca_n_pc + 0.25, pca_cumulative_var[pca_n_pc - 1] - 0.1),
            )
            plt.title("Variance ratios explained by PCs (cumulative)")
            plt.xlabel("PC #")
            plt.ylabel("Explained variance ratio")
            fig.savefig(self._plots_dir / "pca_variance_explained.png", bbox_inches="tight",)
            plt.close("all")

            self.iter_clusters(self.times, pca_n_pc)
            break

    def check_spikes(self, spikes, times):
        """
        Checks for spikes in the data. If none are found, writes a warning and returns.

        Parameters
        ----------
        spikes : ndarray
            Array of spike waveforms.
        times : ndarray
            Array of spike times.

        Returns
        -------
        None
            Writes a warning and returns if no spikes are found.

        """
        if spikes.size == 0:
            (self._data_dir / "no_spikes.txt").write_text(
                "No spikes were found on this channel."
                " The most likely cause is an early recording cutoff."
            )
            warnings.warn("No spikes were found on this channel. The most likely cause is an early recording cutoff.")
            (self._data_dir / "no_spikes.txt").write_text("Sorting finished. No spikes found")
            return

    def iter_clusters(self, spike_times, n_pc):
        """
        Iterates through each cluster, performs GMM-based clustering, and saves the results.

        Parameters
        ----------
        spike_times : ndarray
            1D array of spike times.
        n_pc : int
            Number of principal components to use.

        """
        from cluster import ClusterGMM

        # Be careful with cluster numbers, they are 0-indexed
        tested_clusters = np.arange(self.min_clusters, self.max_clusters)
        clust_results = pd.DataFrame(
            columns=["clusters", "converged", "BIC", "spikes_per_cluster"],
            index=tested_clusters,
        )
        logger.info(f"Testing {len(tested_clusters)} clusters")

        spikes_per_clust = []  # 2, 3, 4 ,5 etc.. clusters
        pbm = ProgressBarManager()
        pbm.init_cluster_bar(len(tested_clusters))
        for num_clust in tested_clusters:
            # if not self.dir_manager.should_process(self.chan_num, num_clust):
            #     continue
            logger.info(f"For {num_clust} in tested_clusters -> {tested_clusters}")
            cluster_data_path = (self._data_dir / f"{num_clust}_clusters")
            cluster_plot_path = (self._data_dir / f"{num_clust}_clusters")
            cluster_data_path.mkdir(parents=True, exist_ok=True)
            cluster_plot_path.mkdir(parents=True, exist_ok=True)

            try:
                model, predictions, bic = ClusterGMM(self.params).fit(
                    self.metrics, num_clust
                )

            except Exception as e:
                logger.warning(f"Error in cluster_gmm: {e}", exc_info=True)
                continue

            if model is None:
                clust_results.loc[num_clust] = [num_clust, bic, False, [0]]
                continue

            # If there are too few waveforms
            # no fmt
            if np.any([
                    len(np.where(predictions[:] == cluster)[0]) <= n_pc + 2
                    for cluster in range(num_clust)
                ]):
                logger.warning(
                    f"There are too few waveforms to properly sort cluster {num_clust + 3}"
                )
                with open(cluster_data_path / "invalid_sort.txt", "w+",) as f:
                    f.write("There are too few waveforms to properly sort this clustering")
                with open(
                    cluster_data_path  / "invalid_sort.txt",
                    "w+",
                ) as f:
                    f.write(
                        "There are too few waveforms to properly sort this clustering"
                    )
                continue

            # Sometimes large amplitude noise interrupts the gmm because the amplitude has
            # been factored out of the scaled spikes. Run through the clusters and find the waveforms that are more than
            # wf_amplitude_sd_cutoff larger than the cluster mean.
            for cluster in range(num_clust):
                logger.info(f"{cluster}")
                this_clust_data = cluster_data_path / f"cluster_{cluster}"
                this_clust_data.mkdir(parents=True, exist_ok=True)

                idx = np.where(predictions[:] == cluster)[0]
                cluster_amplitudes = self.metrics[:, 0][idx]
                cluster_amplitude_mean = np.mean(cluster_amplitudes)
                cluster_amplitude_sd = np.std(cluster_amplitudes)
                reject_wf = np.where(
                    cluster_amplitudes
                    <= cluster_amplitude_mean
                    - self.wf_amplitude_sd_cutoff * cluster_amplitude_sd
                )[0]
                if len(reject_wf) > 0:
                    predictions[reject_wf] = -1

                spikes_per_clust.append(len(idx))
                if len(idx) == 0:
                    continue
                isi, v1ms, v2ms = clust.get_ISI_and_violations(
                    self.times, self.sampling_rate
                )
                cluster_spikes = self.spikes[idx]
                cluster_times = self.times[idx]

                np.save(this_clust_data / "predictions.npy", predictions)
                np.save(this_clust_data / "bic.npy", bic)
                np.save(this_clust_data / "cluster_spikes.npy", cluster_spikes)
                np.save(this_clust_data / "cluster_times.npy", cluster_times)
                np.save(this_clust_data / "cluster_isi.npy", isi)
                np.save(this_clust_data / "cluster_1ms_v.npy", v1ms)
                np.save(this_clust_data / "cluster_2ms_v.npy", v2ms)

            clust_results.loc[num_clust] = [num_clust, True, bic, spikes_per_clust]
            feature_pairs = itertools.combinations(
                list(range(self.metrics.shape[1])), 2
            )
            for f1, f2 in feature_pairs:
                logger.info(f"Plotting {(f1, f2)}")
                fn = "%sVS%s.png" % (self.data_columns[f1], self.data_columns[f2])
                feat_str = f"{self.data_columns[f1]}_vs_{self.data_columns[f2]}"
                feat_str = feat_str.replace(" ", "")
                savename = cluster_plot_path / feat_str
                plot_cluster_features(
                    self.metrics[:, [f1, f2]],
                    predictions,
                    x_label=self.data_columns[f1],
                    y_label=self.data_columns[f2],
                    save_file=str(savename),
                )

            # Plot Mahalanobis distances between cluster pairs
            for this_cluster in range(num_clust):
                savename = self._plots_dir / f"cluster_{this_cluster}" / "mahalonobis_cluster"
                mahalanobis_dist = clust.get_mahalanobis_distances_to_cluster(
                    self.metrics, model, predictions, this_cluster
                )
                title = "Mahalanobis distance to cluster %i" % this_cluster
                plot_mahalanobis_to_cluster(mahalanobis_dist, title, str(savename))
            pbm.update_cluster_bar()

    def superplots(self, maxclust: int):
        """
        Creates superplots for each channel.

        This function conglomerates individual plots into a single, combined plot per channel.

        Parameters
        ----------
        maxclust : int
            The maximum number of clusters to consider for creating superplots.

        Returns
        -------
        None
            This function doesn't return any value; it creates superplots as side effects.

        Raises
        ------
        Exception
            If the superplot cannot be created for a channel, an exception is printed to the console.
        """
        path = self.dir_manager.plots / f"channel_{self.chan_num + 1}"
        outpath = self.dir_manager.plots / f"channel_{self.chan_num + 1}" / "superplots"
        if outpath.exists():
            shutil.rmtree(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        for channel in outpath.glob("*"):
            try:
                currentpath = path + "/" + channel
                os.mkdir(
                    outpath + "/" + channel
                )  # create an output path for each channel
                for soln in range(
                    3, maxclust + 1
                ):  # for each number hpc_cluster solution
                    finalpath = outpath + "/" + channel + "/" + str(soln) + "_clusters"
                    os.mkdir(finalpath)  # create output folders
                    for cluster in range(0, soln):  # for each hpc_cluster
                        mah = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters/Mahalonobis_cluster"
                            + str(cluster)
                            + ".png"
                        )
                        if not np.shape(mah)[0:2] == (480, 640):
                            mah = cv2.resize(mah, (640, 480))
                        wf = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters_waveforms_ISIs/Cluster"
                            + str(cluster)
                            + "_waveforms.png"
                        )
                        if not np.shape(mah)[0:2] == (1200, 1600):
                            wf = cv2.resize(wf, (1600, 1200))
                        isi = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters_waveforms_ISIs/Cluster"
                            + str(cluster)
                            + "_Isolation.png"
                        )
                        if not np.shape(isi)[0:2] == (480, 640):
                            isi = cv2.resize(isi, (640, 480))
                        blank = (
                            np.ones((240, 640, 3), np.uint8) * 255
                        )  # make whitespace for info
                        text = (
                            "Electrode: "
                            + channel
                            + "\nSolution: "
                            + str(soln)
                            + "\nCluster: "
                            + str(cluster)
                        )  # text to output to whitespace (hpc_cluster, channel, and solution numbers)
                        cv2_im_rgb = cv2.cvtColor(
                            blank, cv2.COLOR_BGR2RGB
                        )  # convert to color space pillow can use
                        pil_im = Image.fromarray(cv2_im_rgb)  # get pillow image
                        draw = ImageDraw.Draw(pil_im)  # create draw object for text
                        font = ImageFont.truetype(
                            os.path.split(__file__)[0] + "/bin/arial.ttf", 60
                        )  # use arial font
                        draw.multiline_text(
                            (170, 40), text, font=font, fill=(0, 0, 0, 255), spacing=10
                        )  # draw the text

                        info = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                        im_v = cv2.vconcat([info, mah, isi])
                        im_all = cv2.hconcat([wf, im_v])
                        cv2.imwrite(
                            finalpath + "/Cluster_" + str(cluster) + ".png", im_all
                        )  # save the image
            except Exception as e:
                print(
                    "Could not create superplots for channel "
                    + channel
                    + ". Encountered the following error: "
                    + str(e)
                )

    def compile_isoi(self, maxclust=7, l_ratio_cutoff=0.1):
        """
        Compiles the Inter-Spike Interval (ISI) data for each channel and applies conditional formatting.

        This method iterates through each channel's clustering solutions, reads ISI information,
        and writes the compiled data to an Excel file. It also applies conditional formatting
        based on ISI values.

        Parameters
        ----------
        maxclust : int, optional
            The maximum number of clusters to consider for each channel. Default is 7.
        l_ratio_cutoff : float, optional
            The L-Ratio cutoff value for conditional formatting in the Excel sheet. Default is 0.1.

        Returns
        -------
        None
            Writes compiled ISI data and errors to an Excel file but does not return any value.

        Raises
        ------
        Exception
            If reading ISI information for a channel and solution fails.

        Notes
        -----
        The function does the following:

        1. Iterates through each channel and solution to read ISI information.
        2. Appends ISI data to a DataFrame and writes it to a CSV file.
        3. Creates an Excel file that contains compiled ISI data with conditional formatting.

        """
        path = self.dir_manager.reports / "clusters"
        file_isoi = pd.DataFrame()
        errorfiles = pd.DataFrame(columns=["channel", "solution", "file"])
        for channel in os.listdir(path):
            channel_isoi = pd.DataFrame()
            for soln in range(3, maxclust + 1):
                try:
                    channel_isoi = channel_isoi.append(
                        pd.read_csv(
                            path + "/{}/clusters{}/isoinfo.csv".format(channel, soln)
                        )
                    )
                except Exception as e:
                    print(e)
                    errorfiles = errorfiles.append(
                        [
                            {
                                "channel": channel[-1],
                                "solution": soln,
                                "file": os.path.split(path)[0],
                            }
                        ]
                    )
            channel_isoi.to_csv(
                "{}/{}/{}_iso_info.csv".format(path, channel, channel), index=False
            )  # output data for the whole channel to the proper folder
            file_isoi = file_isoi.append(
                channel_isoi
            )  # add this channel's info to the whole file info
            try:
                file_isoi = file_isoi.drop(columns=["Unnamed: 0"])
            except Exception as e:
                logger.warning(f"{e}")
                pass
        with pd.ExcelWriter(
            os.path.split(path)[0] + f"/{os.path.split(path)[-1]}_compiled_isoi.xlsx",
            engine="xlsxwriter",
        ) as outwrite:
            file_isoi.to_excel(outwrite, sheet_name="iso_data", index=False)
            if (
                errorfiles.size == 0
            ):  # if there are no error csv's add some nans and output to the Excel
                errorfiles = errorfiles.append(
                    [{"channel": "nan", "solution": "nan", "file": "nan"}]
                )
            errorfiles.to_excel(outwrite, sheet_name="errors")
            workbook = outwrite.book
            worksheet = outwrite.sheets["iso_data"]
            redden = workbook.add_format({"bg_color": "red"})
            orangen = workbook.add_format({"bg_color": "orange"})
            yellen = workbook.add_format({"bg_color": "yellow"})
            # add conditional formatting based on ISI's
            worksheet.conditional_format(
                "A2:H{}".format(file_isoi.shape[0] + 1),
                {
                    "type": "formula",
                    "criteria": "=AND($G2>1,$H2>{})".format(str(l_ratio_cutoff)),
                    "format": redden,
                },
            )
            worksheet.conditional_format(
                f"A2:H{file_isoi.shape[0] + 1}",
                {
                    "type": "formula",
                    "criteria": f"=OR(AND($G2>.5,$H2>{str(l_ratio_cutoff)}),$G2>1)",
                    "format": orangen,
                },
            )
            worksheet.conditional_format(
                "A2:H{}".format(file_isoi.shape[0] + 1),
                {
                    "type": "formula",
                    "criteria": f"=OR($G2>.5,$H2>{str(l_ratio_cutoff)})",
                    "format": yellen,
                },
            )
            outwrite.save()
