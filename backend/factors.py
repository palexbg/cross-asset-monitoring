"""Factor lens construction, exposure estimation, and return attribution."""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from backend.config import DataConfig
from .moments import compute_ewma_covar
from .config import (
    FactorDef,
    FactorConfig
)
from .utils import get_returns
from .structs import FactorAnalysisMode, ReturnMethod, Currency

from numba import njit
from typing import List, Optional, Tuple, Union


# TODO:
# *convert the rolling regression (with HAC) to full numba for speed, possibly also using EWMA covariance estimates here in order
#  to be consistent with the factor construction


def triangulate_fx_factor(
    fx_data: pd.DataFrame,
    base_currency: Currency,
    ccy_factor_data: pd.Series
) -> pd.Series:
    """
    Creates a synthetic price series for the Foreign Currency factor relative to 
    the portfolio's base currency, using currency triangulation if necessary.

    Assumes that we have access to UDN prices (G10 vs USD proxy for a foreign currency factor
    UDN is always quoted in dollars, so we just use the currency pair of the base currency to USD
    to triangulate the foreign currency factor if the base currency is not USD.
    """

    # Base Case: USD Portfolio - we do nothing
    if base_currency == Currency.USD:
        return ccy_factor_data.copy()

    # 2. Non-USD Case: Triangulation
    # 1 Unit of Base = X Units of USD
    pair_ticker = f"{base_currency.value}USD=X"

    # Calculate Synthetic Price: (G10 / USD) / (EUR / USD) = G10 / EUR
    # We use bfill() because FX data sometimes has gaps on non-trading days differs from ETFs
    fx_price = fx_data[pair_ticker].reindex(
        ccy_factor_data.index).ffill().bfill()

    synthetic_factor_price = ccy_factor_data.div(fx_price)

    # Rename for clarity in the engine
    synthetic_factor_price.name = "ForeignCurrency"

    return synthetic_factor_price


class FactorConstruction():
    """Construct residualized and optionally scaled factor excess returns.

    Follows the general architecture of https://www.venn.twosigma.com/resources/factor-lens-update
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        factor_definition: List[FactorDef],
        risk_free_rate: pd.Series,
        config: FactorConfig = FactorConfig()
    ):
        self.days_in_year = 252
        self.prices = prices
        self.factor_definition = factor_definition
        self.cov_span = config.cov_span
        self.scale_factors = config.scale_factors
        self.target_yearly_vol = config.target_yearly_vol
        self.target_daily_vol = config.target_yearly_vol / \
            np.sqrt(self.days_in_year)

        self.rf = risk_free_rate.reindex(prices.index).ffill()

        # name to FactorDef mapping
        self.mapping = {f.name: f for f in self.factor_definition}

        # ticker to factor name mapping
        self.ticker_map = {f.name: f.ticker for f in factor_definition}

        # factor name to index mapping for numpy slicing
        self.slicing_map = {name: idx for idx,
                            name in enumerate(self.ticker_map.values())}

    def run(self) -> pd.DataFrame:
        """Build factor excess-return series from raw ETF prices.

        Steps
        -----
        1. Compute 1-day and 5-day excess returns for all raw factor ETFs
           using the configured risk-free rate.
        2. Build an EWMA covariance tensor of 5-day excess returns over
           the ``cov_span`` horizon.
        3. Orthogonalize child factors versus their parents using that
           covariance structure (via ``_orthogonalize``).
        4. Optionally rescale residualized child factors so that their
           realized volatility targets ``target_yearly_vol``.
        5. Add daily log risk-free back to each factor so outputs are
           total-return factor indices suitable for regression.
        """

        # Compute returns
        ret1d_excess, ret5d_excess = self._calc_excess_returns()

        ordered_tickers = list(self.ticker_map.values())

        missing = [t for t in ordered_tickers if t not in ret5d_excess.columns]
        if missing:
            raise ValueError(
                f"Prices input is missing tickers defined in Config: {missing}")

        # Compute ewma covariance matrix of raw factors over time, using excess returns
        data_for_tensor = ret5d_excess[ordered_tickers]
        full_covmat5d = compute_ewma_covar(
            data_for_tensor, span=self.cov_span, annualize=False)

        # Perform orthogonalization
        factors = self._orthogonalize(
            ret1d_excess=ret1d_excess,
            full_covmat=full_covmat5d
        )

        if self.scale_factors:
            factor_vars = compute_ewma_covar(
                factors,
                span=120,  # self.cov_span,
                annualize=True,  # Annualize here to get annualized vol directly
                annualization_factor=self.days_in_year,
                clip_outliers=True
            )

        # scale the residualized factors rollingly (later on we add exponentially)
        # and ALSO add the risk free rate back to get the total
        #   - we add the risk free back to get total returns, which we then store and can use the
        #   - factors for further analyses we have to do this because we are dealing with ETFs here,
        #   - not pure long-short

        for i, f in enumerate(self.factor_definition):
            col = f.name
            if f.parents and self.scale_factors:
                # rolling vol scaling - only on the residualized part, year of vola estimation
                realized_var = factor_vars[:, i, i]
                realized_vol = np.sqrt(realized_var)

                valid_len = len(realized_vol)
                # If there is no valid history for the rolling vol (e.g.
                # too short input series or all values were clipped), skip
                # scaling for this factor to avoid length-mismatch errors.
                if valid_len == 0:
                    continue

                # If all entries are NaN (no usable vol estimates), skip as well.
                n_non_na = int(np.count_nonzero(~np.isnan(realized_vol)))
                if n_non_na == 0:
                    continue

                aligned_index = factors.index[-valid_len:]

                vol_series = pd.Series(
                    realized_vol, index=aligned_index).replace(0.0, np.nan)

                scaler = self.target_yearly_vol / vol_series
                factors[col] = factors[col] * scaler

            # add back risk free to get total returns
            factors[col] = factors[col] + np.log1p(self.rf.values)

        return factors

    def _calc_excess_returns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute 1d and 5d excess returns for raw ETF factor proxies."""

        # Total returns that we need since we operate on ETFs and not pure long-short factors
        ret1d_total = get_returns(
            self.prices, lookback=1, method=ReturnMethod.LOG).fillna(0.0, limit=DataConfig.maxfill_days)
        ret5d_total = get_returns(
            self.prices, lookback=5, method=ReturnMethod.LOG).fillna(0.0, limit=DataConfig.maxfill_days)

        # Excess returns, fix the risk free rate here
        rf_daily = np.log1p(self.rf)
        rf_5d = rf_daily * 5

        ret1d_excess = ret1d_total.sub(rf_daily, axis=0).fillna(
            0.0, limit=DataConfig.maxfill_days)
        ret5d_excess = ret5d_total.sub(rf_5d, axis=0).fillna(
            0.0, limit=DataConfig.maxfill_days)

        # Return Total (for output construction) AND Excess (for beta calculation)
        return ret1d_excess, ret5d_excess

    def _orthogonalize(self, ret1d_excess, full_covmat):
        """Orthogonalize child factors against parents using EWMA covariance.

        For each factor with ``parents`` defined in ``factor_definition``
        we:

        1. Extract parent and child return series from ``ret1d_excess``.
        2. Use the EWMA covariance tensor to form parent-parent and
           parent-child covariance blocks at each time step.
        3. Solve for regression betas of the child on its parents using a
           small ridge for numerical stability.
        4. Store the residual (child minus explained part) as the
           orthogonalized child factor return.

        Top-level factors (no parents) are left unchanged.
        """

        output = pd.DataFrame(
            index=ret1d_excess.index, columns=self.ticker_map.keys())

        for f in self.factor_definition:
            ticker = f.ticker
            name = f.name
            if not f.parents:  # no parents, raw factor as per the hierarchy
                output[name] = ret1d_excess[ticker]
            else:
                parent_indices = [
                    self.slicing_map[self.ticker_map[p]]
                    for p in self.mapping[f.name].parents
                ]

                parent_tickers = [self.ticker_map[p] for p in f.parents]

                target_index = self.slicing_map[f.ticker]

                sigma_xx = np.ascontiguousarray(
                    full_covmat[:, parent_indices][:, :, parent_indices])
                sigma_xy = np.ascontiguousarray(
                    full_covmat[:, parent_indices][:, :, target_index])
                y = ret1d_excess[ticker].to_numpy()
                x = np.ascontiguousarray(
                    ret1d_excess[parent_tickers].to_numpy())

                resid_excess = _residualization_engine(
                    sigma_xx, sigma_xy, y, x
                )

                output[name] = resid_excess

        return output


@njit
def _residualization_engine(
    sigma_xx: np.ndarray,
    sigma_xy: np.ndarray,
    y: np.ndarray,
    x: np.ndarray
) -> np.ndarray:

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


class FactorExposure():
    """Estimate factor betas and decompose portfolio returns into factors."""

    def __init__(self,
                 risk_factors: pd.DataFrame,
                 nav: pd.DataFrame,
                 analysis_mode: FactorAnalysisMode = FactorAnalysisMode.ROLLING,
                 lookback: int = 120,
                 smoothing_window: int = 5,
                 risk_free_rate: Optional[pd.Series] = None,
                 ):
        """Create a factor exposure model for a portfolio and factor set."""

        # params
        self.analysis_mode = analysis_mode
        self.lookback = lookback

        if risk_free_rate is not None:
            self.rf = risk_free_rate.reindex(nav.index).fillna(0.0)
        else:
            self.rf = pd.Series(0.0, index=nav.index)

        rf_daily = np.log1p(self.rf)  # we convert it to log returns here
        # we will use it later on for the 5d smoothed factor exposures
        rf_5d = rf_daily * smoothing_window

        if self.analysis_mode not in [FactorAnalysisMode.ROLLING, FactorAnalysisMode.FULL]:
            raise NotImplementedError(
                'Type must be either FactorAnalysisMode.ROLLING, FactorAnalysisMode.FULL')
        if self.analysis_mode == FactorAnalysisMode.ROLLING and (lookback <= 0 or lookback is None):
            raise ValueError(
                'If type is rolling, lookback has to be an integer bigger than 0')

        # storage
        self.betas = None
        self.t_stats = None
        self.Rsquared = None
        self.resid = None

        # compute 5 days rolling returns to smoothen out potential asynchronicity in factors
        self.Y_excess = get_returns(nav, smoothing_window, method=ReturnMethod.LOG).sub(
            rf_5d, axis=0).fillna(0.0, limit=DataConfig.maxfill_days)
        self.X_excess = get_returns(risk_factors, smoothing_window, method=ReturnMethod.LOG).sub(
            rf_5d, axis=0).fillna(0.0, limit=DataConfig.maxfill_days)

    def run(self):
        """Estimate factor betas on smoothed excess returns.

        In FULL mode, runs a single OLS regression of 5-day portfolio
        excess returns on 5-day factor excess returns using HAC-robust
        standard errors, and broadcasts the resulting betas to the last
        available date.

        In ROLLING mode, walks through time with a ``lookback``-length
        window, re‑estimating the same regression on each window to
        obtain time-varying betas, t‑stats, R-squared and residual
        variance.
        """

        # ['const', 'Equity', 'Rates', ...]
        cols = ['const'] + list(self.X_excess.columns)

        if self.analysis_mode == FactorAnalysisMode.FULL:
            # Static Regression
            betas, t_stats, rsq, resid_var = self._calculate_one_regression(
                self.X_excess, self.Y_excess)

            # Broadcast to full timeframe
            self.betas = pd.DataFrame(
                data=betas.values.reshape(1, -1),
                index=[self.X_excess.index[-1]],
                columns=cols
            )

            self.t_stats = pd.DataFrame(
                data=t_stats.values.reshape(1, -1),
                index=[self.X_excess.index[-1]],
                columns=cols
            )
            self.Rsquared = pd.DataFrame(
                data=rsq,
                index=[self.X_excess.index[-1]],
                columns=['Rsquared']
            )

            self.resid = pd.DataFrame(
                data=resid_var,
                index=[self.X_excess.index[-1]],
                columns=['IdiosyncraticRisk']
            )

        else:  # rolling
            rolling_betas = []
            rolling_tstats = []
            rolling_rsq = []
            rolling_resid = []

            # Use the Excess Returns data
            X_data = self.X_excess
            Y_data = self.Y_excess

            # Loop through time
            for end_idx in range(self.lookback, X_data.shape[0]):
                start_idx = end_idx - self.lookback

                # Slice Window
                X_window = X_data.iloc[start_idx:end_idx, :]
                Y_window = Y_data.iloc[start_idx:end_idx]

                # Check for NaNs (Warmup or Missing Data)
                # If >10% of the window is NaN, skip
                if (X_window.isnull().sum().values > self.lookback * 0.10).any() or \
                   (Y_window.isnull().sum().values > self.lookback * 0.10):

                    # Append NaNs if we skip
                    rolling_betas.append(np.full(len(cols), np.nan))
                    rolling_tstats.append(np.full(len(cols), np.nan))
                    rolling_rsq.append(np.nan)
                    rolling_resid.append(np.nan)
                else:
                    # Run Regression
                    betas, t_stats, rsq, resid_var = self._calculate_one_regression(
                        X_window, Y_window)
                    rolling_betas.append(betas.values)
                    rolling_tstats.append(t_stats.values)
                    rolling_rsq.append(rsq)
                    rolling_resid.append(resid_var)
            self.betas = pd.DataFrame(
                data=np.vstack(rolling_betas),
                index=X_data.index[self.lookback:],
                columns=cols
            ).reindex(X_data.index)

            self.t_stats = pd.DataFrame(
                data=np.vstack(rolling_tstats),
                index=X_data.index[self.lookback:],
                columns=cols
            ).reindex(X_data.index)

            self.Rsquared = pd.DataFrame(
                data=np.array(rolling_rsq),
                index=X_data.index[self.lookback:],
                columns=['Rsquared']
            ).reindex(X_data.index)

            self.resid = pd.DataFrame(
                data=np.array(rolling_resid),
                index=X_data.index[self.lookback:],
                columns=['IdiosyncraticRisk']
            ).reindex(X_data.index)

        return self.betas, self.t_stats, self.Rsquared, self.resid

    def decompose_daily_returns(self, daily_nav: pd.DataFrame, daily_factors: pd.DataFrame, trend_window: int = 126) -> dict:
        """Decompose portfolio returns into daily and rolling factor contributions.

        Procedure
        ---------
        1. Align portfolio NAV, factor indices and betas on a common
           date index.
        2. Compute daily log returns for portfolio and factors and
           subtract the daily log risk-free rate to get excess returns.
        3. Multiply factor excess returns by betas to obtain per-factor
           daily contributions to portfolio excess return; the gap
           between portfolio excess return and total explained excess is
           labeled ``Residual``.
        4. Add explicit ``RiskFree`` and ``Total`` (raw portfolio log
           return) components to the daily output.
        5. Build a ``trend`` view by summing daily log contributions
           over a ``trend_window`` and converting back to simple
           cumulative returns.

        Returns a dictionary with:
              - ``daily``: daily log-return contributions for analysis
              - ``trend``: rolling cumulative simple contributions for
                visualization in heatmaps.
        """

        # align
        common_idx = self.betas.index.intersection(
            daily_nav.index).intersection(daily_factors.index)

        r_f_simple = self.rf.loc[common_idx]
        r_f_log = np.log1p(r_f_simple)

        # beta
        betas = self.betas.loc[common_idx]
        if 'const' in betas.columns:
            betas = betas.drop(columns=['const'])

        r_port_full = get_returns(daily_nav, method=ReturnMethod.LOG)
        r_factors_full = get_returns(daily_factors, method=ReturnMethod.LOG)

        r_port = r_port_full.loc[common_idx]
        r_factors = r_factors_full.loc[common_idx]

        # Excess Returns
        r_port_excess = r_port.sub(r_f_log, axis=0)
        r_factors_excess = r_factors.sub(r_f_log, axis=0)

        # (Beta * Factor Excess Return)
        contribs = r_factors_excess.multiply(betas)

        total_explained_excess = contribs.sum(axis=1)
        specific_selection = r_port_excess - total_explained_excess

        # We explicitly add the rows required for the Venn plot
        daily_output = contribs.copy()
        daily_output['RiskFree'] = r_f_log
        # Selection in terminology of CFA
        daily_output['Residual'] = specific_selection

        daily_output['Total'] = r_port

        # We sum the log returns over the window to get cumulative log return
        trend_log = daily_output.rolling(window=trend_window).sum()

        # Convert to Simple Returns for display
        trend_output = np.exp(trend_log) - 1

        return {
            "daily": daily_output,
            "trend": trend_output
        }

    @staticmethod
    def _calculate_one_regression(X: pd.DataFrame, Y: Union[pd.Series, pd.DataFrame]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Run a single OLS regression with HAC robust errors."""
        X = sm.add_constant(X)
        model = sm.OLS(Y, X, missing='drop')
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        betas = results.params
        t_stats = results.tvalues
        rsq = results.rsquared
        residual_var = results.scale  # variance of the residuals
        return betas, t_stats, rsq, residual_var
