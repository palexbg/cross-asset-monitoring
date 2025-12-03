# Two Sigma factor lens
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from backend.moments import compute_ewma_covar
from backend.config import FactorDef, FACTOR_LENS_UNIVERSE
from backend.utils import get_returns
from numba import njit


@njit
def _residualization_engine(sigma_xx: np.ndarray, sigma_xy: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:

    T = y.shape[0]
    residuals = np.full(T, np.nan)

    # numerical stability for the eigenvalues
    eps = 1e-7
    I = np.eye(sigma_xx.shape[1])

    for t in range(T):
        if np.isnan(sigma_xx[t, 0, 0]):
            residuals[t] = np.nan
            continue
        betas = np.linalg.solve(sigma_xx[t, :, :] + eps * I, sigma_xy[t, :])
        residuals[t] = y[t] - np.dot(betas, x[t])

    return residuals


class FactorEngine():
    """
    Following the architecture of Two Sigma's Factor Lens
    See: https://www.venn.twosigma.com/resources/factor-lens-update
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        config: List[FactorDef],
        cov_span: int = 750
    ):

        self.prices = prices
        self.config = config
        self.cov_span = cov_span

        # name to FactorDef mapping
        self.mapping = {f.name: f for f in self.config}

        # ticker to factor name mapping
        self.ticker_map = {f.name: f.ticker for f in config}

        # factor name to index mapping for numpy slicing
        self.slicing_map = {name: idx for idx,
                            name in enumerate(self.ticker_map.values())}

    def run(self) -> pd.DataFrame:

        # Compute returns
        ret1d, ret5d = self._calc_returns()

        ordered_tickers = list(self.ticker_map.values())

        missing = [t for t in ordered_tickers if t not in ret5d.columns]
        if missing:
            raise ValueError(
                f"Prices input is missing tickers defined in Config: {missing}")

        data_for_tensor = ret5d[ordered_tickers]
        ret1d_subset = ret1d[ordered_tickers]

        # Compute ewma covariance matrix of raw factors over time
        full_covmat = compute_ewma_covar(
            data_for_tensor, span=750, annualize=False)

        # Perform orthogonalization
        factors = self._orthogonalize(
            ret1d=ret1d_subset, full_covmat=full_covmat)

        # scale factors (possibly)

        return factors

    def _calc_returns(self) -> pd.DataFrame:
        ret1d = get_returns(self.prices, lookback=1,
                            type='log').fillna(0.0, limit=5)
        ret5d = get_returns(self.prices, lookback=5,
                            type='log').fillna(0.0, limit=5)
        return ret1d, ret5d

    def _orthogonalize(self, ret1d: pd.DataFrame, full_covmat: np.ndarray) -> pd.DataFrame:

        output = pd.DataFrame(
            index=ret1d.index, columns=self.ticker_map.keys())

        for f in self.config:
            ticker = f.ticker
            name = f.name
            if not f.parents:  # no parents, raw factor as per the hierarchy
                output[name] = ret1d[ticker]
            else:
                parent_indices = [
                    self.slicing_map[self.ticker_map[p]]
                    for p in self.mapping[f.name].parents
                ]

                target_index = self.slicing_map[f.ticker]

                output[name] = _residualization_engine(
                    sigma_xx=np.ascontiguousarray(full_covmat[:,
                                                              parent_indices][:, :, parent_indices]),
                    sigma_xy=np.ascontiguousarray(full_covmat[:, parent_indices
                                                              ][:, :, target_index]),
                    y=ret1d[ticker].to_numpy(),
                    x=np.ascontiguousarray(
                        ret1d.iloc[:, parent_indices].to_numpy())
                )

        return output
