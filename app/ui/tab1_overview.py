from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from backend.perfstats import PortfolioStats
from backend.config import BacktestConfig
from backend.structs import Asset, Currency, RebalPolicies
from backend.utils import (
    get_valid_rebal_vec_dates,
)
from backend.data import YFinanceDataFetcher, UniverseLoader
from backend.backtester import run_backtest


def _get_investment_universe():
    return [Asset(name='SPY', asset_class='Equity', ticker='SPY', description='S&P 500 ETF', currency='USD'),
            Asset(name='IEF', asset_class='Rates', ticker='IEF',
                  currency='USD', description='7-10 Year Treasury ETF'),
            Asset(name='LQD', asset_class='Credit', ticker='LQD',
                  currency='USD', description='Investment Grade Corporate Bond ETF'),
            Asset(name='HYG', asset_class='Credit', ticker='HYG',
                  currency='USD', description='High Yield Corporate Bond ETF'),
            Asset(name='SHY', asset_class='Rates', ticker='SHY',
                  currency='USD', description='1-3 Year Treasury ETF'),
            Asset(name='GLD', asset_class='Commodities',
                  currency='USD', ticker='GLD', description='Gold ETF'),
            Asset(name='BND', asset_class='Bond', ticker='BND',
                  currency='USD', description='Vanguard Total Bond Market ETF'),
            Asset(name='^IRX', asset_class='Rates', ticker='^IRX',
                  description='13 Week Treasury Bill', currency='USD'),
            Asset(name='EL4W.DE', asset_class='Rates', ticker='EL4W.DE', currency='EUR',
                  description='Germany Money Market'),
            Asset(name='CSBGC3.SW', asset_class='Rates', ticker='CSBGC3.SW', currency='CHF',
                  description='Swiss Domestic Short Term Bond')]


def _get_fx_universe():
    return [
        Asset(name="EURUSD=X", asset_class="FX", ticker="EURUSD=X",
              description="EURUSD", currency="USD"),
        Asset(name="CHFUSD=X", asset_class="FX", ticker="CHFUSD=X",
              description="CHFUSD", currency="USD"),
        Asset(name="CHFEUR=X", asset_class="FX", ticker="CHFEUR=X",
              description="CHFEUR", currency="EUR"),
    ]


def _get_factor_universe():
    return [Asset(name='VT', asset_class='EquityFactor', ticker='VT', description='Vanguard Total World Stock ETF', currency='USD'),
            Asset(name='IEF', asset_class='RatesFactor', ticker='IEF',
                  description='iShares 7-10 Year Treasury Bond ETF', currency='USD'),
            Asset(name='LQD', asset_class='CreditFactor', ticker='LQD',
                  description='iShares iBoxx $ Investment Grade Corporate Bond ETF', currency='USD'),
            Asset(name='GSG', asset_class='CommoditiesFactor', ticker='GSG',
                  description='iShares S&P GSCI Commodity-Indexed Trust', currency='USD'),
            Asset(name='VTV', asset_class='ValueFactor', ticker='VTV',
                  description='Vanguard Value Index Fund ETF Shares', currency='USD'),
            Asset(name='MTUM', asset_class='MomentumFactor', ticker='MTUM',
                  description='iShares MSCI USA Momentum Factor ETF', currency='USD'),
            Asset(name='UDN', asset_class='FXFactor', ticker='UDN',
                  description='Adjusted Invesco DB US Dollar Index Bearish Fund', currency='USD')]


def _build_portfolio_universe_table(weights: dict[str, float]) -> pd.DataFrame:
    """Build a table describing the instruments in the current portfolio.

    The table includes ticker, asset class, base currency, weight and description.
    """
    inv_universe = _get_investment_universe()
    inv_map = {a.ticker: a for a in inv_universe}
    rows = []
    for ticker, w in weights.items():
        asset = inv_map.get(ticker)
        if asset is not None:
            rows.append(
                {
                    "Ticker": ticker,
                    "Asset Class": asset.asset_class,
                    "Base Currency": asset.currency,
                    "Weight": f"{w:.1%}",
                    "Description": asset.description,
                }
            )
    return pd.DataFrame(rows)


def _get_asset_class_map() -> dict[str, str]:
    """Return a mapping from ticker to high-level asset class label.

    This is used to aggregate deviations by asset class in the UI.
    """
    inv_universe = _get_investment_universe()
    return {a.ticker: a.asset_class for a in inv_universe}


def _build_key_stats_table(port_stats: PortfolioStats) -> pd.DataFrame:
    """Build a compact stats table from PortfolioStats for display in the UI."""
    stats = port_stats.calculate_stats(mode="basic")

    # Compute annualized return (CAGR) and volatility directly from NAV/returns
    nav = port_stats.nav.dropna()
    rets = port_stats.returns.dropna()
    if not nav.empty and not rets.empty:
        n_days = (nav.index[-1] - nav.index[0]).days
        if n_days > 0:
            cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n_days) - 1.0
        else:
            cagr = float("nan")
        ann_vol = rets.std() * (252 ** 0.5)
    else:
        cagr = float("nan")
        ann_vol = float("nan")

    metric_map = [
        ("Start Period", "Start Period"),
        ("End Period", "End Period"),
        ("Sharpe", "Sharpe"),
        ("Max Drawdown", "Max Drawdown"),
        ("Longest DD Days", "Longest DD Days"),
    ]

    rows = []
    for label, idx in metric_map:
        if idx in stats.index:
            value = stats.loc[idx].iloc[0]
            rows.append({"Metric": label, "Value": value})

    # Custom calendar/rolling period returns from NAV, independent of QuantStats
    if not nav.empty:
        last_nav = nav.iloc[-1]
        last_date = nav.index[-1]

        # YTD: from first trading day of current calendar year to last date
        start_ytd_date = pd.Timestamp(year=last_date.year, month=1, day=1)
        nav_ytd = nav[nav.index >= start_ytd_date]
        if len(nav_ytd) >= 2:
            ytd_ret = nav_ytd.iloc[-1] / nav_ytd.iloc[0] - 1.0
            rows.append({"Metric": "YTD", "Value": f"{ytd_ret:.2%}"})

        # Helper for rolling window returns by trading days
        def _window_ret(window_days: int):
            if len(nav) < 2:
                return None
            # Use trading-day count as an approximation (e.g. 63 ~ 3M)
            start_idx = max(0, len(nav) - window_days)
            sub = nav.iloc[start_idx:]
            if len(sub) >= 2:
                return sub.iloc[-1] / sub.iloc[0] - 1.0
            return None

        three_m = _window_ret(63)
        if three_m is not None:
            rows.append({"Metric": "3M", "Value": f"{three_m:.2%}"})

        six_m = _window_ret(126)
        if six_m is not None:
            rows.append({"Metric": "6M", "Value": f"{six_m:.2%}"})

        one_y = _window_ret(252)
        if one_y is not None:
            rows.append({"Metric": "1Y", "Value": f"{one_y:.2%}"})

    # Append our own annualized metrics (formatted as %)
    if pd.notna(cagr):
        rows.append({"Metric": "Annualized Return (CAGR)",
                    "Value": f"{cagr:.2%}"})
    if pd.notna(ann_vol):
        rows.append({"Metric": "Annualized Volatility",
                    "Value": f"{ann_vol:.2%}"})

    if rows:
        return pd.DataFrame(rows)

    # Fallback: show whatever stats were returned
    return stats.reset_index().rename(columns={"index": "Metric"})


def _clamp_dates_to_index(start_date, end_date, index: pd.DatetimeIndex):
    """Clamp the requested start/end dates to the closest available dates in index.

    This avoids crashes when the user picks dates outside the data range.
    """
    if index.empty:
        return None, None

    idx = index.sort_values()

    # Clamp start: if before min, use first; if after max, also use last
    if pd.to_datetime(start_date) <= idx[0]:
        clamped_start = idx[0]
    elif pd.to_datetime(start_date) >= idx[-1]:
        clamped_start = idx[-1]
    else:
        # nearest date on or after requested start
        pos = idx.get_indexer(
            [pd.to_datetime(start_date)], method="nearest")[0]
        clamped_start = idx[pos]

    # Clamp end similarly
    if pd.to_datetime(end_date) <= idx[0]:
        clamped_end = idx[0]
    elif pd.to_datetime(end_date) >= idx[-1]:
        clamped_end = idx[-1]
    else:
        pos = idx.get_indexer([pd.to_datetime(end_date)], method="nearest")[0]
        clamped_end = idx[pos]

    # Ensure ordering
    if clamped_start > clamped_end:
        clamped_start, clamped_end = clamped_end, clamped_start

    return clamped_start, clamped_end


@st.cache_data(show_spinner=False)
def _build_saa_backtest(
    base_currency: str,
    start_date,
    end_date,
    rebal_policy_name: str,
    weights: dict[str, float],
):
    inv = _get_investment_universe()
    fx = _get_fx_universe()
    factors = _get_factor_universe()

    portfolio_tickers = list(weights.keys())

    data_engine = YFinanceDataFetcher()
    universe_loader = UniverseLoader(data_engine)
    close, risk_free_rate = universe_loader.load_or_fetch_universe(
        close_csv_path=Path("etf_close_prices.csv"),
        investment_universe=inv,
        factor_universe=factors,
        fx_universe=fx,
        base_currency=Currency(base_currency),
        start_date=str(start_date),
        end_date=str(end_date),
    )

    # Clamp requested dates to available price index
    clamped_start, clamped_end = _clamp_dates_to_index(
        start_date, end_date, close.index)
    if clamped_start is None or clamped_end is None:
        return None, None, None

    close = close.loc[clamped_start:clamped_end]

    pf_prices = close[portfolio_tickers].dropna()

    rebal_policy = {
        "US Month Start": RebalPolicies.US_MONTH_START,
        "US Month End": RebalPolicies.US_MONTH_END,
    }[rebal_policy_name]

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

    bt = run_backtest(
        prices=pf_prices,
        target_weights=pf_weights,
        backtest_config=BacktestConfig(),
        rebal_vec=rebal_vec,
    )

    # Also return the clamped date range and underlying prices for UI use
    return bt, risk_free_rate, rebal_vec, pf_prices, clamped_start, clamped_end


def render_overview_tab(
    base_currency: str,
    start_date,
    end_date,
    rebal_policy_name: str,
    w_equity: float
):
    st.title("Portfolio Overview")

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Running backtest..."):
        # For now, build a simple 2-asset weight vector;
        # this can be extended to more assets later.
        weights = {
            "SPY": w_equity,
            "BND": 1.0 - w_equity,
        }

        bt, rf, rebal_vec, pf_prices, eff_start, eff_end = _build_saa_backtest(
            base_currency=base_currency,
            start_date=start_date,
            end_date=end_date,
            rebal_policy_name=rebal_policy_name,
            weights=weights,
        )

    if bt is None:
        st.error(
            "Selected date range has no overlapping data. Please adjust the dates.")
        return

    port_stats = PortfolioStats(backtest_result=bt, risk_free=rf)
    universe_df = _build_portfolio_universe_table(weights)
    custom_stats = _build_key_stats_table(port_stats)

    # Top row: universe table (left) and NAV (right)
    top_left, top_right = st.columns([2, 3])

    with top_left:
        st.subheader("Portfolio universe")
        if not universe_df.empty:
            st.dataframe(
                universe_df.set_index("Ticker"),
                width="stretch",
            )
        else:
            st.caption("No instruments found for current portfolio.")

    with top_right:
        st.subheader("NAV")
        nav = bt.nav.dropna()
        nav_df = nav.reset_index()
        nav_df.columns = ["date", "nav"]
        nav_chart = (
            alt.Chart(nav_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", axis=alt.Axis(format="%b %Y", title=None)),
                y=alt.Y("nav:Q", title="NAV"),
            )
            .properties(height=250)
        )
        st.altair_chart(nav_chart, use_container_width=True)

    # Second row: key stats and weight deviation
    stats_col, weights_col = st.columns([1, 2])

    with stats_col:
        st.subheader("Key stats")
        # Ensure Value column is consistently string-typed to avoid Arrow issues
        stats_display = custom_stats.copy()
        stats_display["Value"] = stats_display["Value"].astype(str)
        st.table(stats_display.set_index("Metric"))

    with weights_col:
        st.subheader("Deviation from target weights")
        # Compute deviation of realized weights vs target per asset.
        realized_w = bt.weights[list(weights.keys())]

        # Identify rebalance dates where a new target is set
        rebal_dates = rebal_vec.index[rebal_vec]
        target_w = pd.DataFrame(0.0, index=realized_w.index,
                                columns=realized_w.columns)

        # For each interval between rebalances, keep the target fixed and
        # measure deviation from that constant target (piecewise constant).
        for i, dt in enumerate(rebal_dates):
            tgt = pd.Series(weights, index=realized_w.columns)
            start = dt
            end = rebal_dates[i + 1] if i + \
                1 < len(rebal_dates) else realized_w.index[-1]
            target_w.loc[start:end, :] = tgt.values

        dev = realized_w - target_w

        # Aggregate deviations by asset class for clearer monitoring
        ticker_to_ac = _get_asset_class_map()
        dev_ac = {}
        for ticker in dev.columns:
            ac = ticker_to_ac.get(ticker, "Other")
            if ac not in dev_ac:
                dev_ac[ac] = dev[ticker].copy()
            else:
                dev_ac[ac] = dev_ac[ac] + dev[ticker]

        dev_ac_df = pd.DataFrame(dev_ac)

        # Build deviation lines by asset class plus a +/-2% corridor
        alert_band = 0.02
        dev_ac_df["Upper corridor"] = alert_band
        dev_ac_df["Lower corridor"] = -alert_band

        # Drop rows where everything (including corridors) is NaN
        dev_ac_df = dev_ac_df.dropna(how="all")
        if not dev_ac_df.empty:
            dev_ac_df = dev_ac_df.copy()
            # Preserve the original datetime index as a proper column
            dev_ac_df["date"] = dev_ac_df.index
            dev_melt = dev_ac_df.melt(
                "date", var_name="series", value_name="value")
            dev_chart = (
                alt.Chart(dev_melt)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("value:Q", title="Deviation from target"),
                    color="series:N",
                )
                .properties(height=250)
            )
            st.altair_chart(dev_chart, use_container_width=True)
        else:
            st.caption("No deviation data available.")

    # Third row: monthly heatmap (left), transaction costs (middle), rolling vol (right)
    # Give rolling volatility a bit more horizontal space so it doesn't look squished.
    col_hm, col_tc, col_rv = st.columns([1, 1, 1.4])

    with col_hm:
        st.subheader("Monthly performance heatmap")
        heatmap_fig = port_stats.plot_monthly_heatmap_fig()
        st.pyplot(heatmap_fig, clear_figure=False)

    with col_tc:
        st.subheader("Transaction costs over time")
        if hasattr(bt, "costs") and bt.costs is not None:
            costs = bt.costs.dropna()
            costs_df = costs.reset_index()
            costs_df.columns = ["date", "costs"]
            costs_chart = (
                alt.Chart(costs_df)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("costs:Q", title="Costs"),
                )
                .properties(height=320)
            )
            st.altair_chart(costs_chart, use_container_width=True)
        else:
            st.caption("No costs series found in backtest result.")

    with col_rv:
        st.subheader("Rolling 6M volatility")
        rv_series = port_stats.get_rolling_vol_series(window=126)
        rv_series = rv_series.dropna()
        if not rv_series.empty:
            rv_df = rv_series.reset_index()
            rv_df.columns = ["date", "vol"]
            rv_chart = (
                alt.Chart(rv_df)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("vol:Q", title="Annualized Vol"),
                )
                .properties(height=320)
            )
            st.altair_chart(rv_chart, use_container_width=True)
        else:
            st.caption("Not enough data to compute rolling volatility.")
