from numba import njit
from backend.utils import freq2days
import numpy as np
import pandas as pd
import pdb

# The core


@njit
def clean_returns_outliers(
    returns: np.ndarray,
    alpha: float,
    initial_vols: np.ndarray,
    clip_treshold: float = 5.0
) -> np.ndarray:
    """
    We use it to clip very high (absolute) returns
    """

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

                # needs to be hooked up on proper demeaning after
                curr_var[n] = alpha*curr_var[n] + (1-alpha)*(r_t**2)

    return clean_returns


@njit
def compute_ewma_kernel(
    returns: np.ndarray,
    alpha: float,
    initial_cov: np.ndarray
) -> np.ndarray:

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
    freq: str = 'B',
    clip_outliers: bool = True
) -> pd.DataFrame:
    # This is related to a multivariate version of IGARCH(1,1), a particular case of GARCH(1, 1)

    data = (returns
            .dropna(axis=0)
            .to_numpy()
            .astype(np.float64)
            )

    alpha = 1.0 - 2.0/(1.0 + span)  # investigate

    T, N = data.shape

    if N == 1:
        data = data.reshape(-1, 1)  # investigate

    warmup = min(span, T)

    init_vols = np.std(data[:warmup, :], axis=0)
    init_cov = np.cov(data[:warmup, :], rowvar=False)

    if clip_outliers:
        data = clean_returns_outliers(
            returns=data, alpha=alpha, initial_vols=init_vols)

    cov_tensor = compute_ewma_kernel(
        returns=data, alpha=alpha, initial_cov=init_cov)

    # nanify the initial warmup period
    cov_tensor[:warmup, :, :] = np.nan

    if annualize:
        ann_factor = freq2days(freq=freq)
        cov_tensor = ann_factor * cov_tensor

    pdb.set_trace()

    return cov_tensor
