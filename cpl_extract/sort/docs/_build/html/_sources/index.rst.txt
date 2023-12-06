=========================
Clustersort Documentation
=========================

Clustersort is heavily adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017) [1]_.

.. note::
    Clustersort is a Python-based framework for processing and analyzing electrophysiological spike data.
    performs waveform extraction and spike sorting. It also provides a number of tools for analyzing
    the resulting spike data. Once spikes are sorted, users are able to post-process the spikes using
    plots for mahalanobis distance, ISI, and autocorrelograms. The data can also be exported to
    a variety of formats for further analysis in other programs.

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   :name: gettingstarted

   Installation <install>
   Guide <guide/index>
   Api <api/index>


Contribute
==========

1. Fork the repository.
2. Create your feature branch.
3. Commit your changes.
4. Open a pull request.

Contact
=======

- GitHub: https://github.com/flynnoconnell/clustersort
- Email: Flynnoconnell@gmail.com

For more in-depth information, consult the API documentation in `docs/api/autosort`.

References
==========

.. [1] Mukherjee, Narendra & Wachutka, Joseph & Katz, Donald. (2017). Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents. 98-105.
