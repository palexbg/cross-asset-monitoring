from numba import njit
import numpy as np
import pandas as pd
import pdb


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
def compute_ewma_mean_kernel(
    returns: np.ndarray,
    alpha: float,
    initial_mean: np.ndarray
) -> np.ndarray:

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
    clip_outliers: bool = True,
    demean: bool = True
) -> pd.DataFrame:
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
        ann_factor = freq2days(freq=freq)
        cov_tensor = ann_factor * cov_tensor

    return cov_tensor


def compute_sample_covar(
    returns: pd.DataFrame,
    window: int,
    annualize: bool = False,
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
        # Simple annualization by frequency, reusing freq2days if available
        from .utils import freq2days  # local import to avoid cycles

        ann_factor = freq2days(freq=freq)
        cov_tensor = ann_factor * cov_tensor

    return cov_tensor
