"""Numba kernels and helpers for return cleaning and covariance estimation.

Implements EWMA-based mean and covariance recursion, simple outlier
clipping on returns, and higher-level helpers to build covariance
tensors used by the risk engines and factor construction.
"""

from numba import njit
import numpy as np
import pandas as pd


@njit
def clean_returns_outliers(
    returns: np.ndarray,
    alpha: float,
    initial_vols: np.ndarray,
    clip_treshold: float = 5.0
) -> np.ndarray:
    """Clip extreme return outliers using an EWMA volatility estimate without look-ahead bias."""

    # Ideas
    # 1) We calculate the EWMA recursively using numba
    # 2) The recursive calculation is needed in order to smoothen out very big deviations (of the type flash crashes/events like covid)
    # so that they do not entirely pollute our volatility estimator

    T, N = returns.shape
    # container to store the cleaned returns
    clean_returns = np.zeros_like(returns)

    # initial mean is 0 (to check the zscore)

    curr_var = initial_vols**2

    for t in range(T):
        for n in range(N):
            r_t = returns[t, n]  # returns at t
            sigma = np.sqrt(curr_var[n])  # curr_var uses data until t-1

            if sigma > 1e-8:
                z_score = r_t/sigma
                if abs(z_score) > clip_treshold:
                    r_t = np.sign(z_score) * clip_treshold * sigma

                clean_returns[t, n] = r_t

                # valid because series were demeaned beforehand
                curr_var[n] = alpha*curr_var[n] + (1-alpha)*(r_t**2)

    return clean_returns


@njit
def compute_ewma_mean_kernel(
    returns: np.ndarray,
    alpha: float,
    initial_mean: np.ndarray
) -> np.ndarray:
    """Numba kernel computing EWMA means for a return matrix."""
    T, N = returns.shape

    mean_history = np.zeros((T, N))

    curr_mean = initial_mean.copy()

    for t in range(T):
        r_t = returns[t, :]

        curr_mean = r_t * (1 - alpha) + alpha * curr_mean

        mean_history[t, :] = curr_mean

    return mean_history


@njit
def compute_ewma_cov_kernel(
    returns: np.ndarray,
    alpha: float,
    initial_cov: np.ndarray
) -> np.ndarray:
    """Numba kernel computing an EWMA covariance tensor over time."""
    T, N = returns.shape

    cov_history = np.zeros((T, N, N))  # that is a simple tensor

    curr_cov = initial_cov.copy()

    for t in range(T):
        r_t = returns[t, :]

        # cov at time t
        instant_cov = np.outer(r_t, r_t)

        curr_cov = instant_cov * (1-alpha) + alpha * curr_cov

        cov_history[t, :, :] = curr_cov

    return cov_history


def compute_ewma_covar(
    returns: pd.DataFrame,
    span: int = 21,
    annualize: bool = False,
    annualization_factor: int = 252,
    clip_outliers: bool = True,
    demean: bool = True
) -> pd.DataFrame:
    """Compute an EWMA covariance tensor for multivariate returns.

    Returns a (T, N, N) NumPy array aligned with the input index, with
    optional outlier clipping, demeaning and annualization.
    """
    # This is related to a multivariate version of IGARCH(1,1), a particular case of GARCH(1, 1)

    data = (returns
            .dropna(axis=0)
            .to_numpy()
            .astype(np.float64)
            )

    alpha = 1.0 - 2.0/(1.0 + span)  # investigate

    T, N = data.shape

    warmup = min(span, T)

    init_mean = np.mean(data[:warmup, :], axis=0)
    init_vols = np.std(data[:warmup, :], axis=0)

    if N == 1:
        init_cov = np.array([[np.var(data[:warmup, :])]])  # cast to 2d
    else:
        init_cov = np.cov(data[:warmup, :], rowvar=False)

    # 0) clip outliers
    if clip_outliers:
        data = clean_returns_outliers(
            returns=data, alpha=alpha, initial_vols=init_vols)

    # 1) demean before computing the variance (more relevant for trending assets)
    if demean:
        ewma_means = compute_ewma_mean_kernel(
            data, alpha=alpha, initial_mean=init_mean)
        data = data - ewma_means

    # 2) calculate the ewma cov
    cov_tensor = compute_ewma_cov_kernel(
        returns=data, alpha=alpha, initial_cov=init_cov)

    # 3) nanify the initial warmup period
    cov_tensor[:warmup, :, :] = np.nan

    # 4) annualization of the covmat, matching the demeaning
    if annualize:
        cov_tensor = annualization_factor * cov_tensor

    return cov_tensor


def compute_sample_covar(
    returns: pd.DataFrame,
    window: int,
    annualize: bool = False,
    annualization_factor: int = 252,
    freq: str = "B"
) -> np.ndarray:
    """Compute rolling sample covariance tensor.

    This uses a simple rolling window sample covariance estimation on the
    input returns DataFrame, producing a (T, N, N) tensor aligned with the
    original index. The first `window`-1 entries are NaN.
    """

    data = returns.dropna(axis=0)
    T, N = data.shape

    cov_tensor = np.full((T, N, N), np.nan, dtype=float)

    # Pre-allocate view for efficiency
    values = data.to_numpy(dtype=float)

    for t in range(window - 1, T):
        window_slice = values[t - window + 1: t + 1]
        cov_t = np.cov(window_slice, rowvar=False)
        cov_tensor[t] = cov_t

    if annualize:
        cov_tensor = annualization_factor * cov_tensor

    return cov_tensor
