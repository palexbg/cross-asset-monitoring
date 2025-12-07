from typing import Union, Optional
import pandas as pd
import numpy as np

from .config import AssetRiskConfig, FactorRiskConfig
from .utils import get_returns
from .moments import compute_ewma_covar, compute_sample_covar
from .structs import CovarianceMethod, ComputeOn


class FactorRiskEngine():
    def __init__(
        self,
        betas: Union[pd.Series, pd.DataFrame],
        factor_prices: Optional[pd.DataFrame] = None,
        residual_var: Optional[pd.Series] = None,
        config: FactorRiskConfig = FactorRiskConfig(),
        rebal_vec: Optional[pd.Series] = None,
        annualize: bool = True,
    ):
        self.betas = betas
        self.factor_prices = factor_prices
        self.residual_var = residual_var
        self.config = config
        self.rebal_vec = rebal_vec
        self.annualize = annualize

        self._returns: Optional[pd.DataFrame] = None
        self._cov_tensor: Optional[np.ndarray] = None

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns is None:
            self._returns = get_returns(
                self.factor_prices,
                method=self.config.returns_method
            ).dropna()
        return self._returns

    @property
    def cov_tensor(self) -> np.ndarray:
        if self._cov_tensor is None:
            if self.config.cov_method == CovarianceMethod.EWMA:
                self._cov_tensor = compute_ewma_covar(
                    returns=self.returns,
                    span=self.config.span,
                    annualize=False
                )
            elif self.config.cov_method == CovarianceMethod.SAMPLE:
                self._cov_tensor = compute_sample_covar(
                    returns=self.returns,
                    window=self.config.span,
                    annualize=False,
                )
            else:
                raise NotImplementedError(
                    f"Covariance method {self.config.cov_method} is not implemented in FactorRiskEngine."
                )
        return self._cov_tensor

    @property
    def latest_covmat(self) -> pd.DataFrame:
        cov = self.cov_tensor[-1]
        cols = self.returns.columns
        return pd.DataFrame(cov, index=cols, columns=cols)

    @staticmethod
    def _drop_const(beta_obj: Union[pd.Series, pd.DataFrame]):
        if isinstance(beta_obj, pd.Series):
            return beta_obj.drop(labels=["const"])
        return beta_obj.drop(columns=["const"])

    @staticmethod
    def _attach_date_index(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        out = df.copy()
        out["Date"] = dt
        out["Factor"] = out.index
        return (
            out.reset_index(drop=True)
            .set_index(["Date", "Factor"])
            .sort_index()
        )

    @staticmethod
    def compute_factor_risk_contribution(
        betas: Union[pd.Series, np.ndarray],
        covmat: Union[pd.DataFrame, np.ndarray],
        annualize: bool = False,
        annualization_factor: int = 252
    ) -> tuple[pd.DataFrame, float]:
        """
        """
        cov = covmat.values if isinstance(covmat, pd.DataFrame) else covmat

        if isinstance(betas, pd.Series):
            b = betas.values
            idx = betas.index
        else:
            b = np.asarray(betas)
            idx = pd.Index(range(len(b)))

        sys_var = b.T @ cov @ b
        sys_vol = np.sqrt(sys_var)

        if sys_vol == 0 or not np.isfinite(sys_vol):
            nan_df = pd.DataFrame({
                "beta": np.full(len(idx), np.nan),
                "mctr": np.full(len(idx), np.nan),
                "ctr": np.full(len(idx), np.nan),
                "ctr_pct": np.full(len(idx), np.nan),
            }, index=idx)
            return nan_df, np.nan

        v = cov @ b
        mctr = v / sys_vol
        ctr = b * mctr
        ctr_pct = ctr / sys_vol

        if annualize:
            scale = np.sqrt(annualization_factor)
            sys_vol = sys_vol * scale
            mctr = mctr * scale
            ctr = ctr * scale
            # ctr_pct unchanged

        out = pd.DataFrame({
            "beta": b,
            "mctr": mctr,
            "ctr": ctr,
            "ctr_pct": ctr_pct
        }, index=idx)

        return out, sys_vol

    def run(self) -> dict:
        B = self._drop_const(self.betas)

        # Factors and betas must align, as we often use warmup period to build factors while betas keep NANs
        B = B.loc[B.index.intersection(self.returns.index)]

        # Latest-only path
        if self.config.compute_on == ComputeOn.LATEST:
            cov_last = self.latest_covmat

            if isinstance(B, pd.Series):
                rc, sys_vol = self.compute_factor_risk_contribution(
                    B, cov_last,
                    annualize=self.annualize,
                    annualization_factor=self.config.annualization_factor
                )
                idio_vol = np.nan
                if self.residual_var is not None:
                    idio_var = self.residual_var.iloc[-1]
                    idio_vol = np.sqrt(idio_var)
                    if self.annualize:
                        # the smoothing window adjustment is needed because the factor betas (and thus the residuals) are smoothed
                        idio_vol *= np.sqrt(self.config.annualization_factor /
                                            self.config.smoothing_window)

                return {
                    "latest_factor_rc": rc,
                    "latest_systematic_vol": sys_vol,
                    "latest_idio_vol": idio_vol
                }

            # over time, we slice the weights vector to be at the rebal dates and calc only those
            last_date = B.index[-1]
            rc, sys_vol = self.compute_factor_risk_contribution(
                B.iloc[-1],
                cov_last,
                annualize=self.annualize,
                annualization_factor=self.config.annualization_factor
            )
            rc = self._attach_date_index(rc, last_date)

            idio_vol = np.nan
            if self.residual_var is not None and last_date in self.residual_var.index:
                idio_var = self.residual_var.loc[last_date]
                idio_vol = np.sqrt(idio_var)
                if self.annualize:
                    idio_vol *= np.sqrt(self.config.annualization_factor /
                                        self.config.smoothing_window)

            return {
                "latest_factor_rc": rc,
                "latest_systematic_vol": sys_vol,
                "latest_idio_vol": idio_vol
            }

        # Over-time path: optionally restrict to rebal dates
        if self.config.compute_on == ComputeOn.REBAL_ONLY and self.rebal_vec is not None:
            B = B.loc[self.rebal_vec]

        cov_dates = self.returns.index
        factors = self.returns.columns

        rows = []
        sys_vol_list = []
        idio_vol_list = []

        for dt in B.index:
            t = cov_dates.get_loc(dt)
            cov_t = pd.DataFrame(
                self.cov_tensor[t], index=factors, columns=factors)

            rc_t, sys_vol_t = self.compute_factor_risk_contribution(
                B.loc[dt],
                cov_t,
                annualize=self.annualize,
                annualization_factor=self.config.annualization_factor
            )
            rc_t = self._attach_date_index(rc_t, dt)
            rows.append(rc_t)

            sys_vol_list.append(sys_vol_t)

            if self.residual_var is not None and dt in self.residual_var.index:
                idio_var = self.residual_var.loc[dt]
                iv = np.sqrt(idio_var)
                if self.annualize:
                    iv *= np.sqrt(self.config.annualization_factor /
                                  self.config.smoothing_window)
                idio_vol_list.append(iv)
            else:
                idio_vol_list.append(np.nan)

        rc_by_date = pd.concat(rows, axis=0).sort_index()

        systematic_vol = pd.Series(
            sys_vol_list, index=B.index, name="SystematicVol"
        )
        idio_vol = pd.Series(
            idio_vol_list, index=B.index, name="IdioVol"
        )

        last_dt = B.index[-1]
        latest_factor_rc = rc_by_date.loc[last_dt].copy()

        return {
            "factor_rc_by_date": rc_by_date,
            "systematic_vol_by_date": systematic_vol,
            "idio_vol_by_date": idio_vol,
            "latest_factor_rc": latest_factor_rc,
            "latest_systematic_vol": systematic_vol.loc[last_dt],
            "latest_idio_vol": idio_vol.loc[last_dt][0] if np.isfinite(idio_vol.loc[last_dt][0]) else np.nan
        }


class AssetRiskEngine():
    def __init__(self,
                 weights: Union[pd.Series, pd.DataFrame],
                 prices: pd.DataFrame,
                 config: AssetRiskConfig = AssetRiskConfig(),
                 rebal_vec: Optional[pd.Series] = None,
                 annualize: bool = True
                 ):
        self.weights = weights
        self.prices = prices
        self.config = config
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
            if self.config.cov_method == CovarianceMethod.EWMA:
                self._cov_tensor = compute_ewma_covar(
                    returns=self.returns,
                    span=self.config.span,
                    annualize=False
                )
            elif self.config.cov_method == CovarianceMethod.SAMPLE:
                self._cov_tensor = compute_sample_covar(
                    returns=self.returns,
                    window=self.config.span,
                    annualize=False,
                )
            else:
                raise NotImplementedError(
                    f"Covariance method {self.config.cov_method} is not implemented in AssetRiskEngine."
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
        return (
            out.reset_index(drop=True)
            .set_index(["Date", "Asset"])
            .sort_index()
        )

    def run(self) -> dict:
        W = self.weights

        # Align weights and returns
        W = W.loc[W.index.intersection(self.returns.index)]

        # Latest-only path
        if self.config.compute_on == ComputeOn.LATEST:
            cov_last = self.latest_covmat
            if isinstance(W, pd.Series):
                risk_contribution, port_vol = self.compute_risk_contribution(
                    W, cov_last,
                    annualize=self.annualize,
                    annualization_factor=self.config.annualization_factor
                )
                return {
                    "latest_rc": risk_contribution,
                    "latest_port_vol": port_vol
                }

            last_date = W.index[-1]
            risk_contribution, port_vol = self.compute_risk_contribution(
                W.iloc[-1], cov_last, annualize=self.annualize, annualization_factor=self.config.annualization_factor)
            risk_contribution = self._attach_date_index(
                risk_contribution, last_date)
            return {
                "latest_rc": risk_contribution,
                "latest_port_vol": port_vol
            }

        # Over-time path: optionally slice weights to rebal dates
        if self.config.compute_on == ComputeOn.REBAL_ONLY and self.rebal_vec is not None:
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
        latest_port_vol = port_vol.loc[last_dt]

        return {
            "rc_by_date": risk_contribution,
            "port_vol_by_date": port_vol,
            "latest_rc": latest_rc,
            "latest_port_vol": latest_port_vol
        }
