import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from backend.perfstats import PortfolioStats
from backend.structs import FactorAnalysisMode
from backend.config import BacktestConfig, AssetRiskConfig
from backend.factors import FactorExposure

from backend.universes import get_investment_universe

from .analysis_context import build_analysis_context
from .theme import (
    FACTOR_COLOR_MAP,
    FACTOR_COLOR_DOMAIN,
    DEFAULT_MISSING_COLOR,
    HEATMAP_SCHEME_RED_YELLOW_GREEN,
)


@st.cache_data(show_spinner=False)
def _cached_factor_attribution(
    nav: pd.Series,
    factors_prices: pd.DataFrame,
    rf: pd.Series,
    lookback: int = 120,
    smoothing_window: int = 5,
    trend_window: int = 126,
):
    """Cached wrapper around FactorExposure + decompose_daily_returns.

    Caches per (nav, factors, rf, lookback, smoothing_window, trend_window)
    configuration so revisiting the tab with the same setup is fast, while any
    change in inputs still triggers a full recomputation.
    """

    nav_df = nav.to_frame("NAV")

    exposure_engine = FactorExposure(
        risk_factors=factors_prices,
        nav=nav_df,
        analysis_mode=FactorAnalysisMode.ROLLING,
        lookback=lookback,
        smoothing_window=smoothing_window,
        risk_free_rate=rf,
    )

    # Run the rolling regression engine once per configuration
    exposure_engine.run()

    # Daily and rolling-window contributions from the exposure engine
    return exposure_engine.decompose_daily_returns(
        nav, factors_prices, trend_window=trend_window
    )


def render_risk_contrib_tab(
    base_currency: str,
    start_date,
    end_date,
    rebal_policy_name: str,
    w_equity: float,
    portfolio_name: str,
):
    st.title(f"Risk & Return Contributions – {portfolio_name}")

    # Top disclaimer (clearly visible)
    st.markdown("---")
    st.info(
        "**Disclaimer:** For educational & demonstrational purposes only. "
        "Loads a local CSV of public prices sourced from Yahoo Finance; not investment advice."
    )

    cfg = BacktestConfig()
    tc_bps = cfg.cost_rate * 10_000.0
    if portfolio_name == "60/40 Buy&Hold":
        rebal_label = "One-time only at beginning"
    else:
        rebal_label = rebal_policy_name

    st.caption(
        f"Portfolio Configuration: {base_currency} | {start_date} → {end_date} | "
        f"Rebal: {rebal_label} | Equity weight: {w_equity:.0%} | "
        f"Costs: {tc_bps:.1f} bps per trade"
    )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Running backtest and risk engines..."):
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
            "Selected date range has no overlapping data. Please adjust the dates.")
        return

    bt = ctx["bt"]
    rf = ctx["rf"]
    factors_prices = ctx.get("factors_prices")

    port_stats = PortfolioStats(backtest_result=bt, risk_free=rf)
    nav = port_stats.nav.dropna()

    if nav.empty:
        st.caption("No NAV series available for the selected configuration.")
        return

    if factors_prices is None or factors_prices.empty:
        st.caption(
            "No factor history available for this configuration – "
            "skipping factor-based return attribution."
        )
    else:
        try:
            # Daily and rolling-window contributions from the exposure engine,
            # cached per configuration for faster tab reloads.
            attr = _cached_factor_attribution(
                nav=nav,
                factors_prices=factors_prices,
                rf=rf,
            )
        except Exception as e:
            attr = None
            st.caption(f"Could not compute factor return attribution: {e}")

        daily_attr = attr.get("daily") if isinstance(attr, dict) else None
        trend_attr = attr.get("trend") if isinstance(attr, dict) else None

        if daily_attr is not None and not daily_attr.empty:
            # Rolling trend heatmap (similar to what we can see here ):
            # https://www.venn.twosigma.com/resources/incorporating-historical-portfolio-analysis-into-your-workflows
            # 6M rolling sum of daily contributions, colored red/green
            # around 0 so positive/negative impact is symmetric.
            st.subheader("Factor contributions to return trend")
            st.caption(
                "Rolling 6M cumulative portfolio return contributions per factor component (including risk-free and residual)."
            )

            # Use the precomputed rolling trend output from the engine.
            heat_attr = trend_attr.copy() if trend_attr is not None else None

            if heat_attr is not None and not heat_attr.empty:
                # Rename for display
                rename_map = {
                    "RiskFree": "Risk-Free Rate",
                }
                heat_attr = heat_attr.rename(columns=rename_map)

                # Drop components that are never present, then restrict to
                # dates where all remaining components are available.
                heat_attr = heat_attr.dropna(axis=1, how="all")

                # Reorder rows: Total on top, then factors (alphabetical),
                # then risk-free and residual at the bottom.
                cols = list(heat_attr.columns)
                ordered_cols = []
                for special in ["Total"]:
                    if special in cols:
                        ordered_cols.append(special)

                other = sorted(
                    [
                        c
                        for c in cols
                        if c not in ["Total", "Risk-Free Rate", "Residual"]
                    ]
                )
                ordered_cols.extend(other)

                for special in ["Risk-Free Rate", "Residual"]:
                    if special in cols and special not in ordered_cols:
                        ordered_cols.append(special)

                heat_attr = heat_attr[ordered_cols]

                # Only keep dates where all remaining components are present.
                heat_attr = heat_attr.dropna(axis=0, how="any")

                if heat_attr.empty:
                    st.caption(
                        "Not enough data to compute a rolling factor contribution trend once all factors are present."
                    )
                else:
                    # Color scale: map minimum contribution to red and maximum
                    # contribution to green, using robust percentiles so the
                    # full palette is used without being dominated by outliers.
                    vals = heat_attr.to_numpy().astype("float64")
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        vmin = float(np.nanpercentile(vals, 3))
                        vmax = float(np.nanpercentile(vals, 97))
                        if vmin == vmax:
                            if vmin == 0.0:
                                vmin, vmax = -1e-6, 1e-6
                            else:
                                vmin *= 0.9
                                vmax *= 1.1
                    else:
                        vmin, vmax = -1e-6, 1e-6

                    trend_long = heat_attr.reset_index().melt(
                        id_vars=heat_attr.index.name or "index",
                        var_name="component",
                        value_name="contrib",
                    )
                    trend_long = trend_long.rename(
                        columns={heat_attr.index.name or "index": "date"}
                    )

                    # Use a discrete date label on the x-axis to avoid any
                    # unintended aggregation that could broadcast the last
                    # date across the entire row.
                    if pd.api.types.is_datetime64_any_dtype(trend_long["date"]):
                        trend_long["date_label"] = trend_long["date"].dt.strftime(
                            "%Y-%m-%d")
                    else:
                        trend_long["date_label"] = trend_long["date"].astype(
                            str)

                    # Add a small helper label so that the tooltip can
                    # distinguish between portfolio 6M return (Total)
                    # and each component's contribution to that return.
                    trend_long["metric_label"] = np.where(
                        trend_long["component"] == "Total",
                        "6M Portfolio Return",
                        "Contribution to 6M return",
                    )

                    heatmap = (
                        alt.Chart(trend_long)
                        .mark_rect()
                        .encode(
                            x=alt.X(
                                "date_label:N",
                                title=None,
                                sort=None,
                            ),
                            y=alt.Y(
                                "component:N",
                                title=None,
                                sort=ordered_cols,
                            ),
                            color=alt.Color(
                                "contrib:Q",
                                title="6M contribution",
                                scale=alt.Scale(
                                    domain=[vmin, vmax],
                                    scheme=HEATMAP_SCHEME_RED_YELLOW_GREEN,
                                    clamp=True,
                                ),
                                legend=alt.Legend(format=".2%"),
                            ),
                            tooltip=[
                                alt.Tooltip("date_label:N", title="Date"),
                                alt.Tooltip("component:N", title="Component"),
                                alt.Tooltip(
                                    "metric_label:N",
                                    title="Metric",
                                ),
                                alt.Tooltip(
                                    "contrib:Q",
                                    title="Value",
                                    format=".4%",
                                ),
                            ],
                        )
                        .properties(height=360)
                    )

                    st.altair_chart(heatmap, width='stretch')
            else:
                st.caption(
                    "Not enough data to compute a rolling factor contribution trend."
                )
        else:
            st.caption(
                "No factor attribution data available for the selected period.")

    # Section 2 & 3: asset and factor risk contributions side by side
    asset_risk = ctx["asset_risk"]
    latest_rc = asset_risk.get("latest_rc") if asset_risk is not None else None

    st.subheader("Risk contributions")
    if latest_rc is not None and not latest_rc.empty:
        try:
            latest_date = latest_rc.index.get_level_values("Date")[-1]
            as_of_str = latest_date.date().isoformat()
            st.caption(
                f"As of {as_of_str} – contributions to portfolio volatility "
                f"by asset, asset class, and factor (via marginal CTR)."
            )
        except Exception:
            st.caption(
                "Contributions to portfolio volatility by asset, asset class, "
                "and factor (via marginal CTR)."
            )
    else:
        st.caption(
            "Contributions to portfolio volatility by asset, asset class, "
            "and factor (via marginal CTR)."
        )

    # Brief clarification of the CTR% metric used in the pies.
    st.caption(
        "**Snapshot Risk Analysis:** These charts show the decomposition of the portfolio's "
        f"*current* estimated volatility (asset-risk and factor-risk decompositions) based on the latest positions and market conditions (EWMA - using {AssetRiskConfig.span} days). "
        "This can differ from the long-term historical realized volatility shown in the overview tab."
    )

    col_asset, col_aclass, col_factor = st.columns(3)

    with col_asset:
        st.markdown("**By asset (holdings)**")
        if latest_rc is None or latest_rc.empty:
            st.caption("No asset (holdings) risk contribution data available.")
        else:
            rc = latest_rc["ctr_pct"].dropna()
            rc_df = rc.rename("ctr_pct").reset_index()
            rc_df.columns = ["asset", "ctr_pct"]

            pie = (
                alt.Chart(rc_df)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("ctr_pct:Q"),
                    color=alt.Color(
                        "asset:N", legend=alt.Legend(title="Asset")),
                    tooltip=[
                        alt.Tooltip("asset:N", title="Asset"),
                        alt.Tooltip("ctr_pct:Q", title="CTR%", format=".1%"),
                    ],
                )
            )

            labels = (
                alt.Chart(rc_df)
                .transform_filter("datum.ctr_pct >= 0.05")
                .mark_text(radius=80, size=11)
                .encode(
                    theta=alt.Theta("ctr_pct:Q"),
                    text=alt.Text("ctr_pct:Q", format=".0%"),
                )
            )

            st.altair_chart(pie.properties(height=320), width='stretch')

            latest_vol = asset_risk.get("latest_port_vol")
            if latest_vol is not None:
                st.markdown(
                    f"**Current Estimated Volatility (Annualized): {latest_vol:.2%}**")
                st.caption("Based on EWMA covariance of recent returns.")

    # Aggregate CTR_% by asset class as defined in the investment universe
    with col_aclass:
        st.markdown("**By asset class (holdings)**")
        if latest_rc is None or latest_rc.empty:
            st.caption("No asset-class risk contribution data available.")
        else:
            inv_universe = get_investment_universe()
            ticker_to_ac = {a.ticker: a.asset_class for a in inv_universe}

            rc_ac = (
                latest_rc["ctr_pct"]
                .groupby(lambda t: ticker_to_ac.get(t, "Other"))
                .sum()
                .dropna()
            )

            rc_ac_df = rc_ac.rename("ctr_pct").reset_index()
            rc_ac_df.columns = ["asset_class", "ctr_pct"]

            # Build a consistent colour scale for asset classes using the
            # shared FACTOR_COLOR_MAP where possible. We use the actual
            # asset-class labels as the domain so no slice disappears; any
            # class that doesn't have an explicit mapping falls back to a
            # neutral grey.
            ac_unique = rc_ac_df["asset_class"].astype(str).unique().tolist()
            ac_domain = ac_unique
            ac_range = [FACTOR_COLOR_MAP.get(
                name, DEFAULT_MISSING_COLOR) for name in ac_domain]
            ac_scale = alt.Scale(domain=ac_domain, range=ac_range)

            pie_ac = (
                alt.Chart(rc_ac_df)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("ctr_pct:Q"),
                    color=alt.Color(
                        "asset_class:N",
                        legend=alt.Legend(title="Asset class"),
                        scale=ac_scale,
                    ),
                    tooltip=[
                        alt.Tooltip("asset_class:N", title="Asset class"),
                        alt.Tooltip("ctr_pct:Q", title="CTR%", format=".1%"),
                    ],
                )
            )

            st.altair_chart(pie_ac.properties(height=320), width='stretch')

    factor_risk = ctx["factor_risk"]

    with col_factor:
        st.markdown("**By factor**")
        if factor_risk is None:
            st.caption("No factor risk contribution data available.")
        else:
            latest_factor_rc = factor_risk.get("latest_factor_rc")
            if latest_factor_rc is not None and not latest_factor_rc.empty:
                rc_f = latest_factor_rc["ctr_pct"].dropna()
                rc_f_df = rc_f.rename("ctr_pct").reset_index()
                rc_f_df.columns = ["factor", "ctr_pct"]

                # Build a consistent colour scale for factors using the
                # shared FACTOR_COLOR_MAP so that, for example, the
                # "Equity" factor matches the "Equity" asset class colour.
                f_unique = rc_f_df["factor"].astype(str).unique().tolist()
                f_domain = [
                    name for name in FACTOR_COLOR_DOMAIN if name in f_unique
                ]
                if f_domain:
                    f_range = [FACTOR_COLOR_MAP[name] for name in f_domain]
                    f_scale = alt.Scale(domain=f_domain, range=f_range)
                else:
                    f_scale = alt.Undefined

                pie_f = (
                    alt.Chart(rc_f_df)
                    .mark_arc(innerRadius=40)
                    .encode(
                        theta=alt.Theta("ctr_pct:Q"),
                        color=alt.Color(
                            "factor:N",
                            legend=alt.Legend(title="Factor"),
                            scale=f_scale,
                        ),
                        tooltip=[
                            alt.Tooltip("factor:N", title="Factor"),
                            alt.Tooltip(
                                "ctr_pct:Q", title="CTR%", format=".1%"
                            ),
                        ],
                    )
                )

                st.altair_chart(pie_f.properties(height=320), width='stretch')

            sys_vol = factor_risk.get("latest_systematic_vol")
            idio_vol = factor_risk.get("latest_idio_vol")
            if sys_vol is not None and idio_vol is not None:
                st.markdown(
                    f"**Decomposition:** Systematic {sys_vol:.2%} | Idiosyncratic {idio_vol:.2%}")
