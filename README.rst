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

.. _install:

============
Installation
============

This pipeline requires Python 3.9+ (SonPy dependency), and numpy <= 1.3.5 (numba dependency).

It is recommended to install using anaconda. 

**Windows:**

.. code-block:: bash

    # Download the Miniconda installer for Windows from the official website
    # https://docs.conda.io/en/latest/miniconda.html

    # Follow the installer instructions

    # Open the Anaconda Prompt and navigate to your project directory
    cd path/to/spike2extract

    # Create and activate the environment
    conda env create -f environment.yml
    conda activate clustersort

    # Install the requirements
    pip install -r requirements.txt
    pip install -e .

**Linux and Intel MacOS:**

.. code-block:: bash

    git clone https://github.com/FlynnOConnell/spike2extract.git
    cd path/to/clustersort

    # Install Miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "${HOME}/miniconda3"
    echo ". ${HOME}/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"

    conda env create -f environment.yml
    conda activate clustersort
    pip install -r requirements.txt
    pip install -e .

**M1 Macs (Apple Silicon):**

.. note::
   M1 Macs require an x86 environment to run certain dependencies. The instructions below guide you through setting this up.

.. code-block:: bash

    # Download the Miniconda installer for MacOSX (x86) from the official website
    # https://docs.conda.io/en/latest/miniconda.html

    # Install Miniconda3 using the x86 architecture
    arch -x86_64 /bin/bash Miniconda3-latest-MacOSX-x86_64.sh -b -p "${HOME}/miniconda3"

    # Add Miniconda3 to your .zshrc or .bash_profile
    echo ". ${HOME}/miniconda3/etc/profile.d/conda.sh" >> ~/.zshrc
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"

    # Create and activate the x86 environment 
    arch -x86_64 conda env create -f environment.yml
    conda activate clustersort

    # Install the requirements
    pip install -r requirements.txt
    pip install -e .

.. warning::
   Ensure you activate the `clustersort` environment before running the pipeline.

Usage
=====

Example notebooks using the pipeline can be found in the notebook (nb) folder.

Recommended processing steps: 

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

To set up your raw data files, move each individual session to its own folder.
To merge two files, simply put both raw data files in the same folder, one with "_pre" and one with "_post" at the end of the filename. Via the shell or GUI, a base directory is chosen that should contain a raw datafile.

Data is stored in a temporary HDF5 file during initialisation, detection and extraction. During clustering, these stores are replaced with .npy files
in the spike_sorting / electrode_# folders.

This h5 file can be opened and inspected using a variety of tools such as `h5pyviewer <https://myhdf5.hdfgroup.org/>`_.