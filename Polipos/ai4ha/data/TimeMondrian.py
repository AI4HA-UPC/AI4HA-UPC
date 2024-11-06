from numpy.random import randint, normal, choice
import numpy as np


def TimeMondrian(n, m, p, q, sigma):
    """
    Generate a time series of Mondrian process.

    Parameters
    ----------
    n : int
        Number of time points.
    m : int
        Number of time series.
    p : float
        Probability of change point.
    q : float
        Probability of change direction.
    sigma : float
        Standard deviation of the noise.

    Returns
    -------
    ndarray
        Time series of Mondrian process.
    """
    x = np.zeros((n, m))
    for i in range(m):
        x[0, i] = normal(0, sigma)
    for i in range(1, n):
        for j in range(m):
            if randint(0, 100) < p * 100:
                x[i, j] = choice([-1, 1]) * x[i - 1, j]
            else:
                x[i, j] = x[i - 1, j]
            if randint(0, 100) < q * 100:
                x[i, j] = -x[i, j]
            x[i, j] += normal(0, sigma)
    return x


def TimeMondrianWithTrend(n, m, p, q, sigma, trend):
    """
    Generate a time series of Mondrian process with trend.

    Parameters
    ----------
    n : int
        Number of time points.
    m : int
        Number of time series.
    p : float
        Probability of change point.
    q : float
        Probability of change direction.
    sigma : float
        Standard deviation of the noise.
    trend : float
        Trend of the time series.

    Returns
    -------
    ndarray
        Time series of Mondrian process with trend.
    """
    x = TimeMondrian(n, m, p, q, sigma)
    for i in range(m):
        x[:, i] += trend * np.arange(n)
    return x


def save_mondrian(p_cp, p_cd, sigma, length, channels, n_samples):
    x = TimeMondrian(length, n_samples * channels, p_cp, p_cd, sigma)
    print(f"Saving TimeMondrian_{length}_{p_cp}_{p_cd}_{sigma}.npy")
    np.save(f"TimeMondrian_{length}_{p_cp}_{p_cd}_{sigma}.npy",
            x.T.reshape(n_samples, channels, length))


if __name__ == "__main__":
     save_mondrian(0.05, 0.05, 1, 1000, 12, 10000)
    # save_mondrian(0.1, 0.1, 2, 1000, 12, 500)
    # save_mondrian(0.2, 0.2, 5, 256, 9, 50000)
    # save_mondrian(0.3, 0.3, 7, 256, 9, 50000)
    # save_mondrian(0.4, 0.4, 9, 256, 9, 50000)
    # save_mondrian(0.5, 0.5, 11, 1000, 12, 50000)
