from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from backend.perfstats import PortfolioStats
from backend.structs import Asset
from backend.config import BacktestConfig
from backend.universes import get_investment_universe

from .analysis_context import build_analysis_context
from .theme import AREA_HIGHLIGHT_COLOR, HEATMAP_SCHEME_RED_YELLOW_GREEN


def _build_portfolio_universe_table(
    weights: dict[str, float],
    weights_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table describing the instruments in the current portfolio.

    The table includes ticker, asset class, base currency, start/end weights
    and description. Start weight is taken from the second NAV date when
    available (fallback to the first), end weight from the last date.
    """
    inv_universe = get_investment_universe()
    inv_map = {a.ticker: a for a in inv_universe}

    # Determine start/end rows for weights, using the 2nd date when possible
    # to avoid all-zeros on the very first date.
    if weights_history is not None and not weights_history.empty:
        idx = weights_history.index
        if len(idx) >= 2:
            start_idx = idx[1]
        else:
            start_idx = idx[0]
        end_idx = idx[-1]

        w_start = weights_history.loc[start_idx]
        w_end = weights_history.loc[end_idx]
    else:
        w_start = pd.Series(weights)
        w_end = pd.Series(weights)

    rows = []
    for ticker, _ in weights.items():
        asset = inv_map.get(ticker)
        if asset is not None:
            ws = float(w_start.get(ticker, 0.0))
            we = float(w_end.get(ticker, 0.0))
            rows.append(
                {
                    "Ticker": ticker,
                    "Asset Class": asset.asset_class,
                    "Base Currency": asset.currency,
                    "Weight start": f"{ws:.1%}",
                    "Weight end": f"{we:.1%}",
                    "Description": asset.description,
                }
            )
    return pd.DataFrame(rows)


def _get_asset_class_map() -> dict[str, str]:
    """Return a mapping from ticker to high-level asset class label.

    This is used to aggregate deviations by asset class in the UI.
    """
    inv_universe = get_investment_universe()
    return {a.ticker: a.asset_class for a in inv_universe}


def _describe_risk_free_proxy(base_currency: str) -> str:
    """Return a human-readable description of the risk-free proxy.

    This mirrors the hard-coded mapping in dailify_risk_free.
    """
    inv_universe = get_investment_universe()
    mapping = {
        "USD": "^IRX",
        "EUR": "EL4W.DE",
        "CHF": "CSBGC3.SW",
    }
    ticker = mapping.get(str(base_currency).upper())
    if ticker is None:
        return f"Risk-free proxy aligned to base currency ({base_currency})."

    asset = next((a for a in inv_universe if a.ticker == ticker), None)
    if asset is None:
        return f"Risk-free proxy: {ticker} (base currency {base_currency})."
    return f"Risk-free proxy: {asset.ticker} — {asset.description} (base currency {base_currency})."


def _monthly_return_matrix(returns: pd.Series) -> pd.DataFrame:
    """
    Convert daily returns to a Year x Month matrix of monthly returns.

    Returns a DataFrame indexed by year with columns as abbreviated month names.
    """
    if returns is None or returns.dropna().empty:
        return pd.DataFrame()

    r = returns.dropna().copy()
    # Limit to business end-of-month aggregation
    monthly = (1.0 + r).resample("M").prod() - 1.0
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    mat = df.pivot(index="year", columns="month", values="ret").sort_index()
    # Convert numeric month to abbreviated month name (Jan, Feb...)
    mat.columns = [pd.Timestamp(2000, m, 1).strftime("%b")
                   for m in mat.columns]
    return mat


def render_overview_tab(
    base_currency: str,
    start_date,
    end_date,
    rebal_policy_name: str,
    w_equity: float,
    portfolio_name: str,
):
    st.title(f"Portfolio Overview – {portfolio_name}")

    # Top disclaimer (clearly visible)
    st.markdown("---")
    st.info(
        "**Disclaimer:** For educational & demonstrational purposes only. "
        "Loads a local CSV of public prices sourced from Yahoo Finance; not investment advice."
    )

    # Data source notice for the portfolio overview
    st.write("Data Source: Yahoo Finance")

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Running backtest, factor lense and risk & return contributions calculations"):
        # For now, build a simple 2-asset weight vector;
        # this can be extended to more assets later.
        weights = {
            "SPY": w_equity,
            "BND": 1.0 - w_equity,
        }

        ctx = build_analysis_context(
            base_currency=base_currency,
            start_date=start_date,
            end_date=end_date,
            rebal_policy_name=rebal_policy_name,
            w_equity=w_equity,
            portfolio_name=portfolio_name,
        )

    if ctx is None:
        st.error(
            "Selected date range has no overlapping data. Please adjust the dates."
        )
        return

    bt = ctx["bt"]
    rf = ctx["rf"]
    rebal_vec = ctx["rebal_vec"]
    pf_prices = ctx["pf_prices"]

    port_stats = PortfolioStats(backtest_result=bt, risk_free=rf)
    universe_df = _build_portfolio_universe_table(weights, bt.weights)
    # Use the backend helper to build a display-ready summary table
    custom_stats = port_stats.summary_table()

    # Top row: universe table (left) and NAV (right)
    top_left, top_right = st.columns([2, 3])

    with top_left:
        st.subheader("Portfolio universe")
        if not universe_df.empty:
            # Show only top 2 rows for the portfolio universe to keep the
            # panel compact; the Key stats panel below will take the
            # remaining vertical space so the two stacked panels match the
            # right column (NAV + Deviation) in total height.
            uni_disp = universe_df.set_index("Ticker").head(2)
            st.dataframe(uni_disp, width="stretch", height=120)
        else:
            st.caption("No instruments found for current portfolio.")

        # Key stats (moved under Portfolio Universe). Render as a taller
        # dataframe so it is not scrollable and visually matches the right
        # column total height (NAV + Deviation = 500 px). We subtract the
        # universe height above (120) to compute the stats height.
        st.subheader("Key stats")
        stats_display = custom_stats.copy()

        def _format_value(row):
            m = str(row.get("Metric", ""))
            v = row.get("Value")
            if pd.isna(v):
                return ""
            # Integers / day counts
            if isinstance(v, (int,)) or (isinstance(v, float) and m.lower().endswith("days")):
                try:
                    return f"{int(v)}"
                except Exception:
                    return str(v)

            # Try numeric conversion for floats represented as strings
            try:
                fv = float(v)
            except Exception:
                return str(v)

            percent_metrics = {
                "YTD",
                "3M",
                "6M",
                "1Y",
                "Annualized Return (CAGR)",
                "Annualized Volatility",
                "Max drawdown",
            }

            if m in percent_metrics or "drawdown" in m.lower() or "return" in m.lower() or "volatility" in m.lower():
                return f"{fv:.2%}"

            return f"{fv:.2f}"

        if not stats_display.empty:
            stats_display["Value"] = stats_display.apply(_format_value, axis=1)
        # Combined desired column height = NAV (250) + Deviation (250) = 500
        stats_height = 500 - 120
        st.dataframe(stats_display.set_index("Metric"),
                     width='stretch', height=stats_height)
        rf_text = _describe_risk_free_proxy(base_currency)
        cfg = BacktestConfig()
        tc_bps = cfg.cost_rate * 10_000.0
        tc_text = (
            f"Transaction costs: {tc_bps:.1f} bps per trade "
            f"(applied to traded notional)."
        )
        st.caption(rf_text)
        st.caption(tc_text)

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
        st.altair_chart(nav_chart, width='stretch')

        # Deviation from target weights (moved under NAV, same height as NAV)
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
            st.altair_chart(dev_chart, width='stretch')
        else:
            st.caption("No deviation data available.")

    # Note: deviation from target weights will be rendered under NAV
    # (same column) so it has the same width/visual prominence as NAV.

    # Third row: split into two main columns. Left column shows monthly
    # heatmap above transaction costs; right column stacks rolling volatility
    # above historical drawdown. This gives the heatmap more horizontal space
    # and groups related diagnostics together.
    # Default heights to ensure variables exist even if branches early-exit
    heatmap_height = 360
    costs_height = 180
    col_left, col_right = st.columns([1, 1])

    # Left: Monthly heatmap (top) and Transaction costs (below)
    with col_left:
        st.subheader("Monthly performance heatmap")
        try:
            r = port_stats.excess_returns.copy()
            if hasattr(port_stats, "nav") and not port_stats.nav.empty:
                r = r.loc[port_stats.nav.index.min(
                ): port_stats.nav.index.max()]
            mat = _monthly_return_matrix(r)

            if mat.empty or mat.shape[0] < 1:
                st.info("Not enough data to compute monthly returns.")
            else:
                # Compute symmetric vmin/vmax around zero so the diverging
                # red/green palette is centered and intuitive.
                try:
                    vmin = float(min(mat.min().min(), 0.0))
                    vmax = float(max(mat.max().max(), 0.0))
                    max_abs = max(abs(vmin), abs(vmax))
                    vmin, vmax = -max_abs, max_abs
                except Exception:
                    vmin, vmax = None, None

                # Render as an Altair tile chart for consistent coloring,
                # tooltips and sizing across environments.
                try:
                    # Prepare long-form dataframe for Altair
                    df_long = mat.reset_index().melt(
                        id_vars=mat.index.name or 'year',
                        var_name='month',
                        value_name='ret',
                    )
                    # Ensure consistent month ordering (Jan..Dec)
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    month_order = [m for m in month_order if m in mat.columns]

                    # Year ordering: most recent first for easier scanning
                    years = sorted(
                        df_long[mat.index.name or 'year'].unique(), reverse=True)

                    # Fallback vmin/vmax
                    if vmin is None or vmax is None:
                        vals = df_long['ret'].dropna().to_numpy()
                        if vals.size:
                            mabs = max(abs(vals.min()), abs(vals.max()))
                            vmin, vmax = -mabs, mabs
                        else:
                            vmin, vmax = -1e-6, 1e-6

                    # Rect tiles
                    rect = (
                        alt.Chart(df_long)
                        .mark_rect()
                        .encode(
                            x=alt.X('month:N', sort=month_order, title=None),
                            y=alt.Y(f"{mat.index.name or 'year'}:O",
                                    sort=years, title=None),
                            color=alt.Color(
                                'ret:Q',
                                scale=alt.Scale(
                                    domain=[vmin, vmax], scheme=HEATMAP_SCHEME_RED_YELLOW_GREEN, clamp=True),
                                legend=alt.Legend(format='.1%'),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    f"{mat.index.name or 'year'}:O", title='Year'),
                                alt.Tooltip('month:N', title='Month'),
                                alt.Tooltip(
                                    'ret:Q', title='Return', format='.2%'),
                            ],
                        )
                    )

                    # Text overlay with 1 decimal percent formatting
                    text = (
                        alt.Chart(df_long)
                        .mark_text(baseline='middle')
                        .encode(
                            x=alt.X('month:N', sort=month_order, title=None),
                            y=alt.Y(f"{mat.index.name or 'year'}:O",
                                    sort=years, title=None),
                            text=alt.Text('ret:Q', format='.1%'),
                            # Use a constant color for text to keep it legible;
                            # could be enhanced with a conditional based on value.
                            color=alt.value('black'),
                        )
                    )

                    heatmap = (alt.layer(rect, text).properties(height=heatmap_height))

                    st.altair_chart(heatmap, width='stretch')
                except Exception:
                    # If altair rendering fails, fall back to styled dataframe
                    styled = mat.style.format("{:.1%}")
                    styled = styled.background_gradient(
                        axis=None, cmap="RdYlGn", vmin=vmin, vmax=vmax)
                    st.dataframe(styled, width='stretch',
                                 height=heatmap_height)
        except Exception as e:
            st.caption(
                f"Could not render monthly performance heatmap (insufficient data or plotting error): {e}"
            )

        # Transaction costs beneath the heatmap
        st.subheader("Transaction costs over time")
        if hasattr(bt, "costs") and bt.costs is not None:
            costs = bt.costs.dropna()
            costs_df = costs.reset_index()
            costs_df.columns = ["date", "costs"]
            costs_height = 180
            costs_chart = (
                alt.Chart(costs_df)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("costs:Q", title="Costs"),
                )
                .properties(height=costs_height)
            )
            st.altair_chart(costs_chart, width='stretch')
        else:
            st.caption("No costs series found in backtest result.")

    # Right: Rolling volatility (top) and historical drawdown (below)
    with col_right:
        st.subheader("Rolling 6M volatility")
        rv_series = port_stats.get_rolling_vol_series(window=126)
        rv_series = rv_series.dropna()
        if not rv_series.empty:
            rv_df = rv_series.reset_index()
            rv_df.columns = ["date", "vol"]
            # Align rolling vol height with the heatmap for visual alignment
            rv_chart = (
                alt.Chart(rv_df)
                .mark_line()
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("vol:Q", title="Annualized Vol"),
                )
                .properties(height=heatmap_height)
            )
            st.altair_chart(rv_chart, width='stretch')
        else:
            st.caption("Not enough data to compute rolling volatility.")

        # Historical drawdown (squeezed below rolling vol)
        st.subheader("Historical drawdown")
        dd_series = port_stats.get_drawdown_series()
        dd_series = dd_series.dropna()
        if not dd_series.empty:
            dd_df = dd_series.reset_index()
            dd_df.columns = ["date", "drawdown"]
            # Make drawdown visually aligned with transaction costs
            dd_chart = (
                alt.Chart(dd_df)
                .mark_area(color=AREA_HIGHLIGHT_COLOR, opacity=0.4)
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(
                        format="%b %Y", title=None)),
                    y=alt.Y("drawdown:Q", title="Drawdown"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("drawdown:Q", title="Drawdown",
                                    format=".1%"),
                    ],
                )
                .properties(height=costs_height)
            )
            st.altair_chart(dd_chart, width='stretch')
        else:
            st.caption("Not enough data to compute drawdowns.")

    # (Duplicate drawdown section removed; drawdown is shown above in the right column)
