from scipy.stats import wasserstein_distance, normal
import numpy as np


def sliced_wasserstein_distance(X, Y, n_slices=10):
    """
    Compute the sliced Wasserstein distance between two data sets

    X and Y are numpy arrays of shape (n_points, n_features) containing the data points
    n_slices is the number of random projections to use

    Random projection vectors are drawn from the standard normal distribution
    """
    d = 0
    for s in range(n_slices):
        projection = normal(size=X.shape[1])
        projection /= np.linalg.norm(projection)
        pX = X @ projection
        pY = Y @ projection
        d += wasserstein_distance(pX, pY)
    return d / n_slices
