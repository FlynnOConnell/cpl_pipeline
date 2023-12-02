"""
Create a Datashader image from an array of waveforms.

This function takes in waveform data and their corresponding x-axis values,
and generates a Datashader image for visualization. It performs downsampling,
DataFrame creation, Datashader canvas setup, aggregation, and image export.
The resulting Datashader image is displayed using Matplotlib.

.. note::
    This function is intended for use with the module.
    It is not intended for general use.
    It relies on the ``datashader`` and ``matplotlib`` packages.

"""
import shutil
from functools import partial

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def waveforms_datashader(waveforms: np.ndarray, x_values: np.ndarray, dir_name="datashader_temp"):
    """
    Create a Datashader image from an array of waveforms.

    This function takes in waveform data and their corresponding x-axis values,
    and generates a Datashader image for visualization. It performs downsampling,
    DataFrame creation, Datashader canvas setup, aggregation, and image export.
    The resulting Datashader image is displayed using Matplotlib.

    Parameters
    ----------
    waveforms : numpy.ndarray
        Numpy array containing waveform data.
    x_values : numpy.ndarray
        Numpy array of x-axis values corresponding to the waveform data.
    dir_name : str, optional
        The directory where temporary files and images are stored.
        Default is "datashader_temp".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the Datashader image.
    ax : matplotlib.axes.Axes
        The Matplotlib axes used for plotting.

    .. note::
        This function is intended for use with the ``spk2py.clustersort`` module.
        It is not intended for general use.
        It relies on the ``datashader`` and ``matplotlib`` packages.

    """

    waveforms = waveforms[:, ::10]
    new_waveforms = np.zeros((waveforms.shape[0], waveforms.shape[1] + 1))
    new_waveforms[:, -1] = np.nan
    new_waveforms[:, :-1] = waveforms

    # Now make an array of x's - the last element is a NaN
    x = np.zeros(x_values.shape[0] + 1)
    x[-1] = np.nan
    x[:-1] = x_values

    # Now make the dataframe
    df = pd.DataFrame(
        {"x": np.tile(x, new_waveforms.shape[0]), "y": new_waveforms.flatten()}
    )

    # Datashader function for exporting the temporary image with the waveforms
    export = partial(export_image, background="white", export_path=dir_name)

    # Use the 5th and 95th percentiles for the y-axis range
    y_min = np.percentile(df["y"], 5)
    y_max = np.percentile(df["y"], 95)

    canvas = ds.Canvas(
        x_range=(np.min(x_values), np.max(x_values)),
        y_range=(y_min, y_max),
        plot_height=1200,
        plot_width=1600,
    )
    agg = canvas.line(df, "x", "y", ds.count())
    export(tf.shade(agg, how="eq_hist"), "tempfile")

    img = imageio.v2.imread(dir_name + "/tempfile.png")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)
    ax.imshow(img)
    ax.set_xticks(np.linspace(0, 1600, 10))
    ax.set_xticklabels(np.floor(np.linspace(np.min(x_values), np.max(x_values), 10)))
    ax.set_yticks(np.linspace(0, 1200, 10))
    ax.set_yticklabels(
        np.floor(np.linspace(df["y"].max() + 10, df["y"].min() - 10, 10))
    )
    del df, waveforms, new_waveforms
    shutil.rmtree(dir_name, ignore_errors=True)
    return fig, ax
