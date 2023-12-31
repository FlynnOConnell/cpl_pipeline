==================
Computational Physiology Pipeline
==================

Electrophyiological Data Analyis Pipeline for use with Spike2 files.

Uses algorithms and code directly from 

.. image:: https://readthedocs.org/projects/cpl-pipeline/badge/?version=latest
    :target: https://cpl-pipeline.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

This primarily relies on the `SonPy <https://github.com/divieira/sonpy/>`_ library for 
extracting data from Spike2 `.smr` files.

This was primarily an attempt to integrate the Spike2 data file type into the Blechpy pipeline,
there are a few issues and features that can be implemented and improved upon.

.. _install:

============
Installation
============

This pipeline requires Python 3.9+ (SonPy dependency), and numpy <= 1.3.5 (numba dependency).

It is recommended to install using miniconda `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

.. code-block:: bash

 if [ ! -d "$HOME/miniconda3" ]; then
        echo "Downloading Miniconda for ARM64..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
        echo "Installing Miniconda..."
        bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    fi
    echo "Initializing Miniconda..."
    $HOME/miniconda3/bin/conda init

    echo "Installing miniconda environment packages..."
    $HOME/miniconda3/bin/conda env create -f $HOME/repos/cpl_pipeline/environment.yml

    $ conda activate cpl_pipeline
    $ pip install cpl_pipeline

.. note::
    Macbook M1 chips also need to configure miniconda to compile binaries for x86_64 CPU architecture because
    SonPy doesn't compile for ARM and has no intentions to.

.. warning::
    Ensure you activate the `cpl_pipeline` environment before running the pipeline.
    Nearly all "Import Error" issues are due to not activating the environment.


Usage
=====

To set up your raw data files, move each individual session to its own folder.
To merge two files, simply put both raw data files in the same folder, one with "_pre" and one with "_post" at the end of the filename. Via the shell or GUI, a base directory is chosen that should contain a raw datafile.

This pipeline can be run from an IPython console, Jupyter notebook, and from the bash shell.

Example notebooks using the pipeline can be found in the notebook (nb) folder.

Recommended processing steps (from Blechpy docs):

.. code-block:: python
    import cpl_pipeline

    dat = cpl_pipeline.Dataset('/path/to/data/dir/') # This will open a dialog box, select the directory/folder containing your .smr file(s)
    dat.initParams(data_quality='hp') # follow GUI prompts. 
    dat.extract_data()          # Extracts raw data into HDF5 store
    dat.create_trial_list()     # Creates table of digital input triggers
    dat.mark_dead_channels()    # View traces and label electrodes as dead, or just pass list of dead channels
    dat.common_average_reference() # Use common average referencing on data. Repalces raw with referenced data in HDF5 store
    dat.detect_spikes()        # Detect spikes in data. Replaces raw data with spike data in HDF5 store
    dat.blech_clust_run(umap=True)       # Cluster data using GMM
    dat.sort_spikes(electrode_number) # Split, merge and label clusters as units. Follow GUI prompts. Perform this for every electrode
    dat.post_sorting() #run this after you finish sorting all electrodes
    dat.make_PSTH_plots() #optional: make PSTH plots for all units 
    dat.make_raster_plots() #optional: make raster plots for all units

.. code-block:: bash
    $ python -m cpl_pipeline --help


Data is stored in a temporary HDF5 file during initialisation, detection and extraction. During clustering, these stores are replaced with .npy files
in the spike_sorting / electrode_# folders.

This h5 file can be opened and inspected using a variety of tools such as `h5pyviewer <https://myhdf5.hdfgroup.org/>`_.