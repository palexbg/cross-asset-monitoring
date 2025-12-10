"""Risk engines for assets and factors.

Provides ``FactorRiskEngine`` and ``AssetRiskEngine`` which turn
returns into covariance tensors and compute systematic/idio vol and
contributions to risk (MCTR, CTR, ctr_pct) across time or at the
latest snapshot.
"""

from typing import Union, Optional
import pandas as pd
import numpy as np

from .config import AssetRiskConfig, FactorRiskConfig
from .utils import get_returns
from .moments import compute_ewma_covar, compute_sample_covar
from .structs import CovarianceMethod, ComputeOn


class FactorRiskEngine():
    """Compute factor-level risk contributions and systematic / idiosyncratic vol."""

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
        """Factor return series used for the risk calculations."""
        if self._returns is None:
            self._returns = get_returns(
                self.factor_prices,
                method=self.config.returns_method
            ).dropna()
        return self._returns

    @property
    def cov_tensor(self) -> np.ndarray:
        """Cached covariance tensor built from factor returns and config."""
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
        """Latest covariance matrix snapshot as a DataFrame."""
        cov = self.cov_tensor[-1]
        cols = self.returns.columns
        return pd.DataFrame(cov, index=cols, columns=cols)

    @staticmethod
    def _drop_const(beta_obj: Union[pd.Series, pd.DataFrame]):
        """Drop intercept column/entry named 'const' from betas object."""
        if isinstance(beta_obj, pd.Series):
            # Drop label if present, otherwise return unchanged
            try:
                return beta_obj.drop(labels=["const"])
            except Exception:
                return beta_obj
        # DataFrame: drop column if it exists, ignore otherwise
        if "const" in beta_obj.columns:
            return beta_obj.drop(columns=["const"])
        return beta_obj

    @staticmethod
    def _attach_date_index(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        """Attach a MultiIndex (Date, Factor) to a factor metrics frame."""
        out = df.copy()
        out["Date"] = dt
        out["Factor"] = out.index
        return (
            out.reset_index(drop=True)
            .set_index(["Date", "Factor"])
            .sort_index()
        )

    def run(self) -> dict:
        """Compute factor risk contributions and systematic/idiosyncratic vol.

        Procedure
        ---------
        1. Align betas with available factor return dates and select the
           dates to evaluate based on ``config.compute_on`` (all,
           rebalance-only or latest).
        2. For each selected date, pull the corresponding covariance
           matrix slice from the cached covariance tensor.
        3. Compute systematic variance as :math:`B^T \Sigma B`, take the
           square root to obtain systematic volatility, and guard against
           zero/non-finite values.
        4. Compute marginal contributions to risk (MCTR) as
           :math:`\Sigma B / \sigma` and contributions to risk (CTR) as
           :math:`B \cdot \text{MCTR}`; ``ctr_pct`` is CTR divided by
           portfolio volatility so that the contributions sum to 1.
        5. If residual variance is provided, convert it to idiosyncratic
           volatility, with optional annualization.

        Returns a dictionary with factor-level risk contributions by
        date plus systematic and idiosyncratic volatility series, and
        convenience "latest" snapshots.
        """
        B = self._drop_const(self.betas)

        # Factors and betas must align, as we often use warmup period to build factors while betas keep NANs
        B = B.loc[B.index.intersection(self.returns.index)]

        # Determine which dates to compute on based on compute_on
        if self.config.compute_on == ComputeOn.LATEST:
            # single latest date
            if isinstance(B, pd.Series):
                B_use = B.to_frame().T
                idx = pd.Index([self.returns.index[-1]])
                B_use.index = idx
            else:
                last_date = B.index[-1]
                B_use = B.loc[[last_date]]
                idx = B_use.index
        else:
            if self.config.compute_on == ComputeOn.REBAL_ONLY and self.rebal_vec is not None:
                B = B.loc[self.rebal_vec]
            B_use = B
            idx = B_use.index

        factors = self.returns.columns
        ret_idx = self.returns.index
        pos = ret_idx.get_indexer(idx)

        cov_t = self.cov_tensor[pos, :, :]
        B_val = B_use.to_numpy(dtype=float)

        # systematic variance per date
        sys_var = np.einsum("ti,tij,tj->t", B_val, cov_t, B_val)
        sys_vol = np.sqrt(sys_var)

        bad = (sys_vol == 0) | ~np.isfinite(sys_vol)
        sys_vol_safe = sys_vol.copy()
        sys_vol_safe[bad] = np.nan

        v = np.einsum("tij,tj->ti", cov_t, B_val)
        mctr = v / sys_vol_safe[:, None]
        ctr = B_val * mctr
        ctr_pct = ctr / sys_vol_safe[:, None]

        if self.annualize:
            scale = np.sqrt(self.config.annualization_factor)
            sys_vol = sys_vol * scale
            mctr = mctr * scale
            ctr = ctr * scale

        if self.residual_var is not None:
            idio_var = self.residual_var.reindex(idx)
            idio_vol = np.sqrt(idio_var)
            if self.annualize:
                idio_vol = idio_vol * np.sqrt(
                    self.config.annualization_factor / self.config.smoothing_window
                )
        else:
            idio_vol = pd.Series(np.nan, index=idx, name="IdioVol")

        rows = []
        for t, dt in enumerate(idx):
            if bad[t]:
                df_t = pd.DataFrame({
                    "beta": np.full(len(factors), np.nan),
                    "mctr": np.full(len(factors), np.nan),
                    "ctr": np.full(len(factors), np.nan),
                    "ctr_pct": np.full(len(factors), np.nan),
                }, index=factors)
            else:
                df_t = pd.DataFrame({
                    "beta": B_val[t],
                    "mctr": mctr[t],
                    "ctr": ctr[t],
                    "ctr_pct": ctr_pct[t],
                }, index=factors)
            df_t = self._attach_date_index(df_t, dt)
            rows.append(df_t)

        rc_by_date = pd.concat(rows, axis=0).sort_index()

        systematic_vol = pd.Series(sys_vol, index=idx, name="SystematicVol")

        last_dt = idx[-1]
        latest_factor_rc = rc_by_date.loc[last_dt].copy()
        latest_systematic = systematic_vol.loc[last_dt]
        # idio_vol.loc[last_dt] may be a scalar (float) or a Series; handle both
        latest_idio = idio_vol.loc[last_dt]
        try:
            latest_idio = float(latest_idio)
        except Exception:
            latest_idio = np.nan
        latest_idio = latest_idio if np.isfinite(latest_idio) else np.nan

        if self.config.compute_on == ComputeOn.LATEST:
            return {
                "latest_factor_rc": latest_factor_rc,
                "latest_systematic_vol": latest_systematic,
                "latest_idio_vol": latest_idio,
            }

        return {
            "factor_rc_by_date": rc_by_date,
            "systematic_vol_by_date": systematic_vol,
            "idio_vol_by_date": idio_vol,
            "latest_factor_rc": latest_factor_rc,
            "latest_systematic_vol": latest_systematic,
            "latest_idio_vol": latest_idio,
        }


class AssetRiskEngine():
    """Compute asset-level risk contributions and portfolio volatility."""

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
        """Asset return series used for risk calculations."""
        if self._returns is None:
            self._returns = get_returns(
                self.prices,
                method=self.config.returns_method
            ).dropna()
        return self._returns

    @property
    def cov_tensor(self) -> np.ndarray:
        """Cached covariance tensor built from asset returns and config."""
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
        """Latest asset covariance matrix snapshot as a DataFrame."""
        cov = self.cov_tensor[-1]
        cols = self.returns.columns
        return pd.DataFrame(cov, index=cols, columns=cols)

    @staticmethod
    def _attach_date_index(df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        """Attach a MultiIndex (Date, Asset) to an asset metrics frame."""
        out = df.copy()
        out["Date"] = dt
        out["Asset"] = out.index
        return (
            out.reset_index(drop=True)
            .set_index(["Date", "Asset"])
            .sort_index()
        )

    def run(self) -> dict:
        """Compute asset risk contributions and portfolio volatility.

        Procedure
        ---------
        1. Select the weight rows to evaluate based on
           ``config.compute_on`` (all dates, rebalance-only, or latest)
           and align them with the available return index.
        2. For each selected date, pull the matching covariance matrix
           slice from the cached covariance tensor.
        3. Compute portfolio variance as :math:`w^T \Sigma w`, take the
           square root to get portfolio volatility, and mask out
           zero/non-finite values.
        4. Compute marginal contributions to risk (MCTR) as
           :math:`\Sigma w / \sigma` and contributions to risk (CTR) as
           :math:`w \cdot \text{MCTR}`; ``ctr_pct`` is CTR divided by
           portfolio volatility so that buckets sum to 100% of risk.
        5. Package per-date risk-contribution DataFrames together with
           a time series of portfolio volatility and a convenient
           "latest" snapshot for dashboards.
        """
        W = self.weights

        # Determine which dates to compute on based on compute_on
        if self.config.compute_on == ComputeOn.LATEST:
            cov_last = self.latest_covmat
            if isinstance(W, pd.Series):
                W_use = W.to_frame().T
                idx = pd.Index([self.returns.index[-1]])
                W_use.index = idx
            else:
                W_df = W.loc[W.index.intersection(self.returns.index)]
                last_date = W_df.index[-1]
                W_use = W_df.loc[[last_date]]
                idx = W_use.index
        else:
            if isinstance(W, pd.Series):
                raise ValueError(
                    "For over-time computation, weights must be a DataFrame")
            W = W.loc[W.index.intersection(self.returns.index)]
            if self.config.compute_on == ComputeOn.REBAL_ONLY and self.rebal_vec is not None:
                W = W.loc[self.rebal_vec]
            W_use = W
            idx = W_use.index

        ret_idx = self.returns.index
        pos = ret_idx.get_indexer(idx)

        cov_t = self.cov_tensor[pos, :, :]
        W_val = W_use.to_numpy(dtype=float)

        # portfolio variance per date
        port_var = np.einsum("ti,tij,tj->t", W_val, cov_t, W_val)
        port_vol = np.sqrt(port_var)

        bad = (port_vol == 0) | ~np.isfinite(port_vol)
        port_vol_safe = port_vol.copy()
        port_vol_safe[bad] = np.nan

        v = np.einsum("tij,tj->ti", cov_t, W_val)
        mctr = v / port_vol_safe[:, None]
        ctr = W_val * mctr
        ctr_pct = ctr / port_vol_safe[:, None]

        if self.annualize:
            scale = np.sqrt(self.config.annualization_factor)
            port_vol = port_vol * scale
            mctr = mctr * scale
            ctr = ctr * scale

        cols = W_use.columns
        rc_list = []
        for t, dt in enumerate(idx):
            if bad[t]:
                df_t = pd.DataFrame({
                    "weight": np.full(len(cols), np.nan),
                    "mctr": np.full(len(cols), np.nan),
                    "ctr": np.full(len(cols), np.nan),
                    "ctr_pct": np.full(len(cols), np.nan),
                }, index=cols)
            else:
                df_t = pd.DataFrame({
                    "weight": W_val[t],
                    "mctr": mctr[t],
                    "ctr": ctr[t],
                    "ctr_pct": ctr_pct[t],
                }, index=cols)
            df_t = self._attach_date_index(df_t, dt)
            rc_list.append(df_t)

        risk_contribution = pd.concat(rc_list, axis=0).sort_index()

        port_vol_series = pd.Series(
            port_vol,
            index=idx,
            name="PortfolioVolatility",
        )

        last_dt = idx[-1]
        latest_rc = risk_contribution.loc[last_dt].copy()
        latest_port_vol = port_vol_series.loc[last_dt]

        if self.config.compute_on == ComputeOn.LATEST:
            return {
                "latest_rc": latest_rc,
                "latest_port_vol": latest_port_vol,
            }

        return {
            "rc_by_date": risk_contribution,
            "port_vol_by_date": port_vol_series,
            "latest_rc": latest_rc,
            "latest_port_vol": latest_port_vol,
        }
