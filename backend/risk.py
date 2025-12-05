from typing import Union, Optional
import pandas as pd
import numpy as np

from .config import RiskConfig
from .utils import get_returns
from .moments import compute_ewma_covar


class FactorRiskEngine():
    pass


class AssetRiskEngine():
    def __init__(self,
                 weights: Union[pd.Series, pd.DataFrame],
                 prices: pd.DataFrame,
                 config: RiskConfig = RiskConfig(),
                 compute_over_time: bool = False,
                 rebal_vec: Optional[pd.Series] = None,
                 annualize: bool = True
                 ):
        self.weights = weights
        self.prices = prices
        self.config = config
        self.compute_over_time = compute_over_time
        self.rebal_vec = rebal_vec
        self.annualize = annualize

        self._returns: Optional[pd.DataFrame] = None
        self._cov_tensor: Optional[np.ndarray] = None

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns is None:
            self._returns = get_returns(
                self.prices,
                method=self.config.returns_method
            ).dropna()
        return self._returns

    @property
    def cov_tensor(self) -> np.ndarray:
        if self._cov_tensor is None:
            self._cov_tensor = compute_ewma_covar(
                returns=self.returns,
                span=self.config.span,
                annualize=False
            )
        return self._cov_tensor

    @property
    def latest_covmat(self) -> pd.DataFrame:
        cov = self.cov_tensor[-1]
        cols = self.returns.columns
        return pd.DataFrame(cov, index=cols, columns=cols)

    @staticmethod
    def compute_risk_contribution(
        weights: Union[np.ndarray, pd.Series],
        covmat: Union[np.ndarray, pd.DataFrame],
        annualize: bool = False,
        annualization_factor: int = 252
    ) -> tuple[pd.DataFrame, float]:

        cov = covmat.values if isinstance(covmat, pd.DataFrame) else covmat

        if isinstance(weights, pd.Series):
            w = weights.values
            idx = weights.index
        else:
            w = np.asarray(weights)
            idx = pd.Index(range(len(w)))

        port_vol = np.sqrt(w.T @ cov @ w)

        # This is a degenerate case when we have missing values
        if port_vol == 0 or not np.isfinite(port_vol):
            nan_df = pd.DataFrame({
                "weight": np.full(len(idx), np.nan),
                "mctr": np.full(len(idx), np.nan),
                "ctr": np.full(len(idx), np.nan),
                "ctr_pct": np.full(len(idx), np.nan),
            }, index=idx)
            return nan_df, np.nan

        mctr = (cov @ w) / port_vol
        ctr = w * mctr
        ctr_pct = ctr / port_vol

        if annualize:
            scale = np.sqrt(annualization_factor)
            port_vol = port_vol * scale
            mctr = mctr * scale
            ctr = ctr * scale

        out = pd.DataFrame({
            "weight": w,
            "mctr": mctr,
            "ctr": ctr,
            "ctr_pct": ctr_pct
        }, index=idx)

        return out, port_vol

    @staticmethod
    def _attach_date_index(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        out = df.copy()
        out["Date"] = dt
        out["Asset"] = out.index
        return out.reset_index(drop=True).set_index(["Date", "Asset"]).sort_index()

    def run(self) -> dict:
        W = self.weights

        # just using latest
        if not self.compute_over_time:
            cov_last = self.latest_covmat
            if isinstance(W, pd.Series):
                risk_contribution, port_vol = self.compute_risk_contribution(
                    W, cov_last,
                    annualize=self.annualize,
                    annualization_factor=self.config.annualization_factor
                )
                return {
                    "latest_rc": risk_contribution,
                    "latest_vol": port_vol
                }

            last_date = W.index[-1]
            risk_contribution, port_vol = self.compute_risk_contribution(
                W.iloc[-1], cov_last, annualize=self.annualize, annualization_factor=self.config.annualization_factor)
            risk_contribution = self._attach_date_index(
                risk_contribution, last_date)
            return {
                "latest_rc": risk_contribution,
                "latest_vol": port_vol
            }

        # over time, we slice the weights vector to be at the rebal dates and calc only those
        if self.rebal_vec is not None:
            W = W.loc[self.rebal_vec]

        risk_contribution_over_time = []
        port_vol_over_time = []

        for i, dt in enumerate(W.index):
            t = self.returns.index.get_loc(dt)
            cov_t = self.cov_tensor[t, :, :]

            rc_t, vol_t = self.compute_risk_contribution(
                W.loc[dt],
                cov_t,
                annualize=self.annualize,
                annualization_factor=self.config.annualization_factor)

            rc_t = self._attach_date_index(rc_t, dt)
            risk_contribution_over_time.append(rc_t)
            port_vol_over_time.append(vol_t)

        risk_contribution = (
            pd.concat(risk_contribution_over_time, axis=0)
            .sort_index()
        )

        port_vol = pd.Series(
            port_vol_over_time,
            index=W.index,
            name="PortfolioVolatility"
        )

        last_dt = W.index[-1]
        latest_rc = risk_contribution.loc[last_dt].copy()
        latest_vol = float(port_vol.loc[last_dt])

        return {
            "rc_by_date": risk_contribution,
            "port_vol_by_date": port_vol,
            "latest_rc": latest_rc,
            "latest_vol": latest_vol
        }
