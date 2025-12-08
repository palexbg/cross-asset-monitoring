# Two Sigma factor lens
import pandas as pd
import numpy as np
import statsmodels.api as sm

from .moments import compute_ewma_covar
from .config import FactorDef, FactorConfig
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
    """
    Following the architecture of Two Sigma's Factor Lens
    See: https://www.venn.twosigma.com/resources/factor-lens-update
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

        # Compute returns
        ret1d_total, ret1d_excess, ret5d_excess = self._calc_returns()

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
            ret1d_total=ret1d_total,
            ret1d_excess=ret1d_excess,
            full_covmat=full_covmat5d
        )
        # scale the residualized factors rollingly (later on we add exponentially)
        # That way the betas later on are comparable and interpretable

        for f in self.factor_definition:
            col = f.name
            if f.parents and self.scale_factors:
                rolling_vol = pd.Series(factors[col]).rolling(window=60).std()
                scaler = self.target_daily_vol / rolling_vol
                factors[col] = factors[col] * scaler

            # we add the risk free back to get total returns, which we then store and can use the factors for further analyses
            # we have to do this because we are dealing with ETFs here, not pure long-short
            factors[col] = factors[col] + np.log1p(self.rf.values)

        return factors

    def _calc_returns(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Total returns that we need since we operate on ETFs and not pure long-short factors
        ret1d_total = get_returns(
            self.prices, lookback=1, method=ReturnMethod.LOG).fillna(0.0, limit=5)
        ret5d_total = get_returns(
            self.prices, lookback=5, method=ReturnMethod.LOG).fillna(0.0, limit=5)

        # Excess returns, fix the risk free rate here
        rf_daily = np.log1p(self.rf)
        rf_5d = rf_daily * 5

        ret1d_excess = ret1d_total.sub(rf_daily, axis=0).fillna(0.0, limit=5)
        ret5d_excess = ret5d_total.sub(rf_5d, axis=0).fillna(0.0, limit=5)

        # Return Total (for output construction) AND Excess (for beta calculation)
        return ret1d_total, ret1d_excess, ret5d_excess

    def _orthogonalize(self, ret1d_excess, full_covmat):

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
    def __init__(self,
                 risk_factors: pd.DataFrame,
                 nav: pd.DataFrame,
                 analysis_mode: FactorAnalysisMode = FactorAnalysisMode.ROLLING,
                 lookback: int = 120,
                 smoothing_window: int = 5,
                 risk_free_rate: Optional[pd.Series] = None,
                 ):
        # params
        self.analysis_mode = analysis_mode
        self.lookback = lookback

        if risk_free_rate is not None:
            self.rf = risk_free_rate.reindex(nav.index).fillna(0.0)
        else:
            self.rf = pd.Series(0.0, index=nav.index)

        rf_daily = np.log1p(self.rf)
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
            rf_5d, axis=0).fillna(0.0, limit=5)
        self.X_excess = get_returns(risk_factors, smoothing_window, method=ReturnMethod.LOG).sub(
            rf_5d, axis=0).fillna(0.0, limit=5)

    def run(self):

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

    def decompose_daily_returns(self, daily_nav: pd.DataFrame, daily_factors: pd.DataFrame) -> pd.DataFrame:

        # Align
        common_idx = self.betas.index.intersection(
            daily_nav.index).intersection(daily_factors.index)
        r_f_simple = self.rf.loc[common_idx]              # simple daily RF
        r_f_log = np.log1p(r_f_simple)                  # daily log RF

        betas = self.betas.loc[common_idx]
        if 'const' in betas.columns:
            betas = betas.drop(columns=['const'])

        # Get Daily Excess Returns

        r_port = get_returns(daily_nav.loc[common_idx])
        r_port_excess = r_port.sub(r_f_log, axis=0)

        r_factors = get_returns(daily_factors.loc[common_idx])
        r_factors_excess = r_factors.sub(r_f_log, axis=0)

        # contribution
        contribs = r_factors_excess.multiply(betas)

        # Alpha = R_port_excess - Sum(Factor_Contribs)
        total_explained = contribs.sum(axis=1)
        alpha = r_port_excess - total_explained

        # 4. Compile Result
        output = contribs.copy()
        output['RiskFree'] = r_f_log
        output['Alpha'] = alpha
        output['Total_Modeled'] = total_explained + r_f_log + alpha
        output['Total_Actual'] = r_port

        return output

    @staticmethod
    def _calculate_one_regression(X: pd.DataFrame, Y: Union[pd.Series, pd.DataFrame]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X, missing='drop')
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        betas = results.params
        t_stats = results.tvalues
        rsq = results.rsquared
        residual_var = results.scale  # variance of the residuals
        return betas, t_stats, rsq, residual_var
