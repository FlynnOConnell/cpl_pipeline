"""
helpers.py

"""
import logging

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture


def cluster_gmm(data, n_clusters, n_iter, restarts, threshold):
    """
    Cluster the data using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : array-like
        The input data to be clustered, generally the output of a PCA, as a 2-D array.
    n_clusters : int
        The number of clusters to use in the GMM.
    n_iter : int
        The maximum number of iterations to perform in the GMM.
    restarts : int
        The number of times to restart the GMM with different initializations.
    threshold : float
        The convergence threshold for the GMM.

    Returns
    -------
    best_fit_gmm : object
        The best-fitting GMM object.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.
    min_bayesian : float
        The minimum Bayesian information criterion (BIC) value achieved across all restarts.
    """

    g = []
    bayesian = []

    # -------- Run the GMM ------------ #
    try:
        for i in range(restarts):
            g.append(
                GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    tol=threshold,
                    random_state=i,
                    max_iter=n_iter,
                )
            )
            g[-1].fit(data)
            if g[-1].converged_:
                bayesian.append(g[-1].bic(data))
            else:
                del g[-1]

        bayesian = np.array(bayesian)
        best_fit = np.where(bayesian == np.min(bayesian))[0][0]

        predictions = g[best_fit].predict(data)
        return g[best_fit], predictions, np.min(bayesian)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error in clusterGMM: {e}", exc_info=True)


def get_lratios(data, predictions):
    """
    Calculate L-ratios for each cluster, a measure of cluster quality.

    Parameters
    ----------
    data : array-like
        The input data, generally the output of a PCA, as a 2-D array.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.

    Returns
    -------
    Lrats : dict
        A dictionary with cluster labels as keys and the corresponding L-ratio values as values.
    """
    lrats = {}
    df = np.shape(data)[1]
    for ref_cluster in np.sort(np.unique(predictions)):
        if ref_cluster < 0:
            continue
        ref_mean = np.mean(data[np.where(predictions == ref_cluster)], axis=0)
        ref_covar_i = np.linalg.inv(
            np.cov(data[np.where(predictions == ref_cluster)], rowvar=False)
        )
        ls = [
            1 - chi2.cdf((mahalanobis(data[point, :], ref_mean, ref_covar_i)) ** 2, df)
            for point in np.where(predictions[:] != ref_cluster)[0]
        ]
        lratio = sum(ls) / len(np.where(predictions == ref_cluster)[0])
        lrats[ref_cluster] = lratio
    return lrats
