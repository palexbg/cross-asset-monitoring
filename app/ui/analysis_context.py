"""This is an orchestrator for building the analysis context shared across UI tabs."""
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.config import (
    BacktestConfig,
    FACTOR_LENS_UNIVERSE,
    AssetRiskConfig,
    FactorRiskConfig,
)
from backend.data import YFinanceDataFetcher, UniverseLoader
from backend.factors import FactorConstruction, FactorExposure
from backend.risk import AssetRiskEngine, FactorRiskEngine
from backend.structs import (
    Currency,
    FactorAnalysisMode,
    RebalPolicies,
    ReturnMethod,
    ComputeOn,
)
from backend.universes import (
    get_factor_universe,
    get_investment_universe,
    get_fx_universe,
)
from backend.utils import (
    build_index_from_returns,
    get_valid_rebal_vec_dates,
)
from backend.backtester import run_backtest


def _resolve_rebal_policy(name: str) -> RebalPolicies:
    """An exception to handle the special mapping we have for the rebalancing policies and the buy&hold portfolio"""
    mapping = {
        "US Month Start": RebalPolicies.US_MONTH_START,
        "US Month End": RebalPolicies.US_MONTH_END,
    }
    # For the 60/40 Buy&Hold preset we locate an initial starting trade date at beginnign of the month
    if name == "One-time only at beginning":
        return RebalPolicies.US_MONTH_START

    return mapping[name]


@st.cache_data(show_spinner=False)
def build_analysis_context(
        base_currency: str,
        start_date,
        end_date,
        rebal_policy_name: str,
        w_equity: float,
    portfolio_name: str,
):
    """
    Build and cache all analysis objects shared across tabs.

    This orchestrates the existing backend components without adding
    new engine logic. It is intentionally kept in the UI layer.
    """

    inv = get_investment_universe()
    fx = get_fx_universe()
    factors_meta = get_factor_universe()

    weights = {"SPY": w_equity, "BND": 1.0 - w_equity}
    portfolio_tickers = list(weights.keys())

    data_engine = YFinanceDataFetcher()
    universe_loader = UniverseLoader(data_engine)

    close, risk_free_rate = universe_loader.load_or_fetch_universe(
        close_csv_path=Path("cached_etf_close_prices.csv"),
        investment_universe=inv,
        factor_universe=factors_meta,
        fx_universe=fx,
        base_currency=Currency(base_currency),
        start_date=str(start_date),
        end_date=str(end_date),
    )

    if close.empty:
        return None

    # Clamp requested dates to available price index
    idx = close.index.sort_values()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    start = max(start, idx[0])
    end = min(end, idx[-1])
    if start > end:
        return None

    close = close.loc[start:end]
    pf_prices = close[portfolio_tickers].dropna()

    rebal_policy = _resolve_rebal_policy(rebal_policy_name)
    valid_dates, rebal_vec = get_valid_rebal_vec_dates(
        schedule=rebal_policy,
        price_index=pf_prices.index,
    )

    pf_weights = pd.DataFrame(
        index=pf_prices.index,
        columns=portfolio_tickers,
        data=0.0,
    )
    for ticker, w in weights.items():
        pf_weights.loc[valid_dates, ticker] = w

    # Trading schedule: for standard SAA and custom allocations we
    # rebalance on each schedule date; for the 60/40 Buy&Hold preset
    # we only trade once at the first valid rebalance date and then
    # let weights drift.
    trade_vec = rebal_vec.copy()
    if portfolio_name == "60/40 Buy&Hold":
        if trade_vec.any():
            first_date = trade_vec[trade_vec].index[0]
            trade_vec.loc[:] = False
            trade_vec.loc[first_date] = True

    bt = run_backtest(
        prices=pf_prices,
        target_weights=pf_weights,
        backtest_config=BacktestConfig(),
        rebal_vec=trade_vec,
    )

    # Asset-level risk: we only need the latest contributions for the
    # UI pies, so compute on the last date only for all presets.
    asset_cfg = AssetRiskConfig(compute_on=ComputeOn.LATEST)

    asset_risk = AssetRiskEngine(
        weights=bt.weights,
        prices=pf_prices,
        config=asset_cfg,
        rebal_vec=trade_vec,
        annualize=True,
    ).run()

    # Factor construction (same plumbing as Tab 2)
    factor_tickers = [f.ticker for f in factors_meta]
    factor_prices_raw = close[factor_tickers].dropna()

    factor_engine = FactorConstruction(
        prices=factor_prices_raw,
        factor_definition=FACTOR_LENS_UNIVERSE,
        risk_free_rate=risk_free_rate,
    )

    factors_ret = factor_engine.run()
    factors_prices = build_index_from_returns(
        factors_ret, method=ReturnMethod.LOG
    )

    nav = bt.nav.dropna()
    common_idx = nav.index.intersection(factors_prices.index)
    if common_idx.empty:
        return {
            "bt": bt,
            "rf": risk_free_rate,
            "rebal_vec": trade_vec,
            "pf_prices": pf_prices,
            "factors_prices": pd.DataFrame(),
            "betas": None,
            "t_stats": None,
            "rsq": None,
            "resid": None,
            "asset_risk": asset_risk,
            "factor_risk": None,
        }

    factors_aligned = factors_prices.loc[common_idx]
    nav_aligned = nav.loc[common_idx].to_frame("NAV")

    exposure_engine = FactorExposure(
        risk_factors=factors_aligned,
        nav=nav_aligned,
        analysis_mode=FactorAnalysisMode.ROLLING,
        lookback=120,
        smoothing_window=5,
        risk_free_rate=risk_free_rate,
    )
    betas, t_stats, rsq, resid = exposure_engine.run()

    factor_risk = None
    if betas is not None and resid is not None:
        try:
            # Factor risk: same logic as asset risk – compute only the
            # latest point, since the UI consumes a single snapshot.
            factor_cfg = FactorRiskConfig(compute_on=ComputeOn.LATEST)

            factor_risk = FactorRiskEngine(
                betas=betas,
                factor_prices=factors_aligned,
                residual_var=resid,
                config=factor_cfg,
                rebal_vec=trade_vec,
                annualize=True,
            ).run()
        except Exception:
            factor_risk = None

    return {
        "bt": bt,
        "rf": risk_free_rate,
        "rebal_vec": trade_vec,
        "pf_prices": pf_prices,
        "factors_prices": factors_aligned,
        "betas": betas,
        "t_stats": t_stats,
        "rsq": rsq,
        "resid": resid,
        "asset_risk": asset_risk,
        "factor_risk": factor_risk,
    }
