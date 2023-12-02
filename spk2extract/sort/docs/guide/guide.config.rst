.. _config-module:

Configuration Guide
===================

Overview
--------

This class manages configurations for the clustersort pipeline. It reads from an INI-style configuration file and provides methods to access configurations for various sections.

.. moduleauthor:: Flynn O'Connell


Initialize a new ``SpkConfig`` object by specifying the ``cfg_path`` parameter. If no path is provided, it defaults to a predefined location.

Attributes
----------

- ``cfg_path``: Path to the configuration file, either provided by the user or a default path.
- ``config``: A ``ConfigParser`` object containing the loaded configurations.
- ``params``: A dictionary containing all the configuration parameters.

Methods
-------

get_section(section: str)
    Returns a dictionary containing key-value pairs for the given section.

set(section: str, key: str, value: Any)
    Sets a value for a configuration parameter within a specified section.

get_all()
    Returns a dictionary containing all key-value pairs from all sections.


Sections
--------

.. _run-section:

run
~~~
Configuration parameters for the runtime of the pipeline.

    .. _run-resort-limit-key:

    - resort-limit
        - The maximum number of times the sorting process can be rerun.
        - Default: 3

    .. _run-cores-used-key:

    - cores-used
        - The number of cores to be used during the run.
        - Default: 8

    .. _run-weekday-run-key:

    - weekday-run
        - The number of runs allowed on a weekday.
        - Default: 2

    .. _run-weekend-run-key:

    - weekend-run
        - The number of runs allowed on a weekend.
        - Default: 8

    .. _run-run-type-key:

    - run-type
        - Defines the type of run (Auto/Manual).
        - Default: Auto

    .. _run-manual-run-key:

    - manual-run
        - The number of manual runs allowed.
        - Default: 2

.. _path-section:

path
~~~~
Here we define various paths necessary for the script, set by default to subdirectories in the parent directory of the specified path.

    .. _path-data-path-key:

    - data
        - Path to the directory where the data to be processed is located.
        - Default: None specified

.. _cluster-section:

cluster
~~~~~~~
Parameters defining the clustering process:

    .. _cluster-max-clusters-key:

    - max-clusters
        - Maximum number of clusters to use in the clustering algorithm.
        - Default: 7

    .. _cluster-max-iterations-key:

    - max-iterations
        - Maximum number of iterations for the gaussian mixture model. This is fed into scipis GMM.
            See here: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
            This is typically anywhere from 100 to 10000, with 1000 being a typical starting point.
        - Default: 1000

    .. _cluster-convergence-criterion-key:

    - convergence-criterion
        - The criterion for convergence in the clustering algorithm.
        - Default: .0001

    .. _cluster-random-restarts-key:

    - random-restarts
        - Number of random restarts in the clustering process to avoid local minima. This is typically a value between
            1 and 10. Because the GMM is stochastic, the results will vary slightly each time and is very sensitive
            to the random seed. It is recommended to start at 1000, and increase if the results are not satisfactory.
        - Default: 10

    .. _cluster-l-ratio-cutoff-key:

    - l-ratio-cutoff
        - The cutoff value for the L-Ratio metric, used to assess cluster quality.
        - Default: .1

.. _breach-section:

breach
~~~~~~
Parameters involved in signal preprocessing and spike detection:

    .. _breach-disconnect-voltage-key:

    - disconnect-voltage
        - Voltage level that indicates a disconnection in the signal, to detect noise or artifacts.
        - Default: 1500

    .. _breach-max-breach-rate-key:

    - max-breach-rate
        - The maximum rate at which breaches (potentially signal artifacts or spikes) can occur before it is considered noise.
        - Default: .2

    .. _breach-max-breach-count-key:

    - max-breach-count
        - The maximum count of breaches allowed in a given window of time.
        - Default: 10

    .. _breach-max-breach-avg-key:

    - max-breach-avg
        - Perhaps the average breach level over a defined window.
        - Default: 20

    .. _breach-intra-hpc_cluster-cutoff-key:

    - intra-hpc_cluster-cutoff
        - A cutoff value for considering a signal as noise based on some intra-cluster metric.
        - Default: 3

.. _filter-section:

filter
------
Filtering parameters to isolate the frequency range of interest:

    .. _filter-low-cutoff-key:

    - low-cutoff
        - The low cutoff frequency for a band-pass filter.
        - Default: 600

    .. _filter-high-cutoff-key:

    - high-cutoff
        - The high cutoff frequency for the band-pass filter.
        - Default: 3000

.. _spike-section:

spike
~~~~~
Spike detection and extraction parameters:

    .. _spike-pre-time-key:

    - pre-time
        - Time before a spike event to include in each spike waveform, in seconds.
        - Default: .2

    .. _spike-post-time-key:

    - post-time
        - Time after a spike event to include in each spike waveform, in seconds.
        - Default: .6

    .. _spike-sampling-rate-key:

    - sampling-rate
        - The sampling rate of the recording, in Hz.
        - Default: 20000

.. _detection-section:

detection
---------
Standard deviation parameters for spike detection and artifact removal:

    .. _detection-spike-detection-key:

    - spike-detection
        - A multiplier for the standard deviation of the noise to set a threshold for spike detection.
        - Default: 2.0

    .. _detection-artifact-removal-key:

    - artifact-removal
        - A threshold for artifact removal, based on a multiple of the standard deviation.
        - Default: 10.0

.. _pca-section:

pca
~~~
Parameters defining how principal component analysis (PCA) is conducted on the spike waveforms:

    .. _pca-variance-explained-key:

    - variance-explained
        - The proportion of variance explained to determine the number of principal components to retain.
        - Default: .95

    .. _pca-use-percent-variance-key:

    - use-percent-variance
        - Whether to use percent variance to determine the number of components to retain. If 0, use all components.
        - Default: 1

    .. _pca-principal-component-n-key:

    - principal-component-n
        - An alternative to variance-explained, specifying the number of principal components to retain directly.
        - Default: 5

.. _postprocess-section:

postprocess
~~~~~~~~~~~
Post-processing parameters:

    .. _postprocess-reanalyze-key:

    - reanalyze
        - Whether to reanalyze the data.
        - Default: 0

    .. _postprocess-simple-gmm-key:

    - simple-gmm
        - Whether to use a simple Gaussian Mixture Model in the post-processing.
        - Default: 1

    .. _postprocess-image-size-key:

    - image-size
        - The size of images generated during post-processing.
        - Default: 70

    .. _postprocess-temporary-dir-key:

    - temporary-dir
        - The directory to store temporary files during processing.
        - Default: user's home directory followed by '/tmp_python'
        - Note: This directory is deleted after processing is complete.

INI Configuration File
----------------------

This file is the easiest entrypoint to change parameters. You can specify where this file
is created with the ``cfg_path`` attribute.

- ``run``: Contains runtime settings like ``resort-limit``, ``cores-used``.
- ``path``: Contains path settings like directories for ``run``, ``results``.
- ``cluster``: Contains clustering parameters like ``max-clusters``, ``max-iterations``.
- ``breach``: Contains breach analysis parameters like ``disconnect-voltage``, ``max-breach-rate``.
- ``filter``: Contains filter parameters like ``low-cutoff``, ``high-cutoff``.
- ``spike``: Contains spike-extraction settings like ``pre-time``, ``post-time``.

Note: All values are stored as strings due to the nature of INI files. It's up to the user to convert these to appropriate types.

Example
-------

.. code-block:: python

    cfg = SpkConfig()
    run = cfg.run
    print(type(run), run)

    cfg.set('run', 'resort-limit', 5)
    print(cfg.run['resort-limit'])

See Also
--------

- `configparser from python std library <https://docs.python.org/3/library/configparser.html>`_
