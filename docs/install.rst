.. _install:

============
Installation
============

For a quick start, see :ref:`quickstart <quickstart>`.

This pipeline requires Python 3.9+, and numpy <= 1.3.5 to comply with numba restrictions.

* `cpl_pipeline on GitHub <https://github.com/FlynnOConnell/cpl_pipeline/>`_

It is recommended to install using `mambaforge <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_ this will drastically speed up environment creation:

Installing from source
======================

**Linux and MacOS:**

.. code-block:: bash

    git clone https://github.com/FlynnOConnell/spike2extract.git
    cd path/to/cpl_pipeline
    # This is for MambaForge, but you can use conda if you want
    # Note if you use conda, but want to go the mamba route, you will really want to uninstall miniconda/anaconda first
    wget -O Mambaforge.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge.sh -b -p "${HOME}/conda"
    # !!!! FOR BASH USERS !!!!
    # If you dont know what these are, then use this one
    # If you use zsh, just change this to ~/.zshrc
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    source "${HOME}/conda/etc/profile.d/conda.sh"


If you're getting ``conda: command not found``, you need to add ``conda`` to your path.
Look in your home directory, you should have a mambaforge or miniforge3 folder, depending on
your method of installation. Add that folder/bin to your path:
`export PATH="/home/username/mambaforge/bin:$PATH"`

.. code-block:: bash

    mamba env create -f environment.yml # this will take a while
    conda activate cpl_pipeline
    pip install -r requirements.txt
    pip install -e .

Additionally, though not recommended, ``cpl_pipeline`` can be installed directly from pip:

.. warning::
   pip installing has **not** been tested on systems other than linux.
   Using ``mamba`` has been tested on each platform.
   As has docker.

.. code-block:: bash

    pip install cpl_pipeline