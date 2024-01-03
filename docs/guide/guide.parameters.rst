run

Configuration parameters for the runtime of the pipeline.

        resort-limit
                The maximum number of times the sorting process can be rerun.
                Default: 3

        cores-used
                The number of cores to be used during the run.
                Default: 8

        weekday-run
                The number of runs allowed on a weekday.
                Default: 2

        weekend-run
                The number of runs allowed on a weekend.
                Default: 8

        run-type
                Defines the type of run (Auto/Manual).
                Default: Auto

        manual-run
                The number of manual runs allowed.
                Default: 2

path

Here we define various paths necessary for the script, set by default to subdirectories in the parent directory of the specified path.

        data
                Path to the directory where the data to be processed is located.
                Default: None specified

cluster

Parameters defining the clustering process:

        max-clusters
                Maximum number of clusters to use in the clustering algorithm.
                Default: 7

        max-iterations

                Maximum number of iterations for the gaussian mixture model. This is fed into scipis GMM.
                    See here: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html This is typically anywhere from 100 to 10000, with 1000 being a typical starting point.

                Default: 1000

        convergence-criterion
                The criterion for convergence in the clustering algorithm.
                Default: .0001

        random-restarts

                Number of random restarts in the clustering process to avoid local minima. This is typically a value between
                    1 and 10. Because the GMM is stochastic, the results will vary slightly each time and is very sensitive to the random seed. It is recommended to start at 1000, and increase if the results are not satisfactory.

                Default: 10

        l-ratio-cutoff
                The cutoff value for the L-Ratio metric, used to assess cluster quality.
                Default: .1

breach

Parameters involved in signal preprocessing and spike detection:

        disconnect-voltage
                Voltage level that indicates a disconnection in the signal, to detect noise or artifacts.
                Default: 1500

        max-breach-rate
                The maximum rate at which breaches (potentially signal artifacts or spikes) can occur before it is considered noise.
                Default: .2

        max-breach-count
                The maximum count of breaches allowed in a given window of time.
                Default: 10

        max-breach-avg
                Perhaps the average breach level over a defined window.
                Default: 20

        intra-hpc_cluster-cutoff
                A cutoff value for considering a signal as noise based on some intra-cluster metric.
                Default: 3

filter

Filtering parameters to isolate the frequency range of interest:

        low-cutoff
                The low cutoff frequency for a band-pass filter.
                Default: 600

        high-cutoff
                The high cutoff frequency for the band-pass filter.
                Default: 3000

spike

Spike detection and extraction parameters:

        pre-time
                Time before a spike event to include in each spike waveform, in seconds.
                Default: .2

        post-time
                Time after a spike event to include in each spike waveform, in seconds.
                Default: .6

        sampling-rate
                The sampling rate of the recording, in Hz.
                Default: 20000

detection

Standard deviation parameters for spike detection and artifact removal:

        spike-detection
                A multiplier for the standard deviation of the noise to set a threshold for spike detection.
                Default: 2.0

        artifact-removal
                A threshold for artifact removal, based on a multiple of the standard deviation.
                Default: 10.0

pca

Parameters defining how principal component analysis (PCA) is conducted on the spike waveforms:

        variance-explained
                The proportion of variance explained to determine the number of principal components to retain.
                Default: .95

        use-percent-variance
                Whether to use percent variance to determine the number of components to retain. If 0, use all components.
                Default: 1

        principal-component-n
                An alternative to variance-explained, specifying the number of principal components to retain directly.
                Default: 5

postprocess

Post-processing parameters:

        reanalyze
                Whether to reanalyze the data.
                Default: 0

        simple-gmm
                Whether to use a simple Gaussian Mixture Model in the post-processing.
                Default: 1

        image-size
                The size of images generated during post-processing.
                Default: 70

        temporary-dir
                The directory to store temporary files during processing.
                Default: user's home directory followed by '/tmp_python'
                Note: This directory is deleted after processing is complete.

