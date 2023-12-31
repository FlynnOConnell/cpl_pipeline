"""
Create a Datashader image from an array of waveforms.

This function takes in waveform data and their corresponding x-axis values,
and generates a Datashader image for visualization. It performs downsampling,
DataFrame creation, Datashader canvas setup, aggregation, and image export.
The resulting Datashader image is displayed using Matplotlib.

"""
import datashader as ds
import datashader.transfer_functions as tf
# from datashader.utils import export_image
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def waveforms_datashader(waveforms, threshold=None):
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
    threshold : float, optional
        Threshold value for plotting a horizontal line on the image.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the Datashader image.
    ax : matplotlib.axes.Axes
        The Matplotlib axes used for plotting.

    """

    if waveforms.shape[0] == 0:
        return None

    # Make a pandas dataframe with two columns, x and y, holding all the data.
    # The individual waveforms are separated by a row of NaNs.
    # First downsample the waveforms 10 times (to remove the effects of 10 times upsampling during de-jittering)
    waveforms = waveforms[:, ::10]
    x_values = np.arange(len(waveforms[0])) + 1

    lower_percentile = 5
    upper_percentile = 95
    y_lower = np.percentile(waveforms, lower_percentile)
    y_upper = np.percentile(waveforms, upper_percentile)

    # Then make a new array of waveforms - the last element of each waveform is a NaN
    new_waveforms = np.zeros((waveforms.shape[0], waveforms.shape[1] + 1))
    new_waveforms[:, -1] = np.nan
    new_waveforms[:, :-1] = waveforms

    # Now make an array of x's - the last element is a NaN
    x = np.zeros(x_values.shape[0] + 1)
    x[-1] = np.nan
    x[:-1] = x_values

    # Now make the dataframe
    df = pd.DataFrame({"x": np.tile(x, new_waveforms.shape[0]), "y": new_waveforms.flatten()})

    # Produce a datashader canvas
    canvas = ds.Canvas(
        x_range=(np.min(x_values), np.max(x_values)),
        y_range=(y_lower, y_upper),
        plot_height=1200,
        plot_width=1600,
    )

    # Aggregate the data
    agg = canvas.line(df, "x", "y", ds.count())
    # Transfer the aggregated data to image using log transform and export the temporary image file
    img = tf.shade(agg, how="eq_hist")
    img = tf.set_background(img, "white")

    # Figure sizes chosen so that the resolution is 100 dpi
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)

    # Start plotting
    ax.imshow(img.to_pil())
    # Set ticks/labels - 10 on each axis
    ax.set_xticks(np.linspace(0, 1600, 10))
    ax.set_xticklabels(np.floor(np.linspace(np.min(x_values), np.max(x_values), 10)))
    ax.set_yticks(np.linspace(0, 1200, 10))
    yticklabels = np.floor(np.linspace(df["y"].max() + 10, df["y"].min() - 10, 10))
    ax.set_yticklabels(yticklabels)

    if threshold is not None and y_lower <= threshold <= y_upper:
        scaled_thresh = (threshold - y_upper) * (1200 / (y_lower - y_upper))
        ax.axhline(scaled_thresh, linestyle="--", color="r", alpha=0.3)

    # Delete the dataframe
    del df, waveforms, new_waveforms

    # Return and figure and axis for adding axis labels, title and saving the file
    return fig, ax
