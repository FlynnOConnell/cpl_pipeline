.. currentmodule:: clustersort

.. _api-doc:

API Documentation
=================

DirectoryManager
----------------

The clustersort pipeline manages directories for 2 primary purposes:

- an entrypoint for the data being sorted
- a place to store plots and temporary data files

Both of these use cases are handled via the :class:`~clustersort.DirectoryManager`.

.. admonition:: Example

    .. code-block:: python

        from clustersort import DirectoryManager as dm
        from clustersort import SpkConfig as config
        # create a directory manager
        dm = dm('path/to/data', 'path/to/output')

        # create a configuration object
        config = config()

        # set the parameters
        config.n_channels = 32
        config.n_features = 3
        config.n_clusters = 5

        # save the configuration
        config.save(dm.config_path)

.. autoclass:: clustersort.DirectoryManager
    :members:
    :noindex:

SpkConfig
---------------------

For the actual sorting, there are a variety of paramters that can be tuned based on the type
of data the user is working with. This is managed by :class:`~clustersort.SpkConfig`.
These parameters are covered in detail in the :ref:`configuration guide <config-module>`.

.. autoclass:: clustersort.SpkConfig
    :members:
    :noindex:
