from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

from backend.structs import FactorAnalysisMode, ReturnMethod
from backend.config import BacktestConfig
from backend.factors import FactorExposure
from backend.config import FACTOR_LENS_UNIVERSE
from backend.universes import get_factor_universe
from backend.utils import get_returns

from .analysis_context import build_analysis_context
from .theme import (
    FACTOR_COLOR_MAP,
    FACTOR_COLOR_DOMAIN,
    RANGE_COLORS,
    DIVERGING_SCHEME_BLUE_ORANGE,
    DEFAULT_MISSING_COLOR,
)


def _build_latest_beta_table(
    risk_factors: pd.DataFrame,
    nav: pd.Series,
    risk_free_rate: pd.Series,
) -> pd.DataFrame:
    nav_df = nav.to_frame("NAV")

    exposure = FactorExposure(
        risk_factors=risk_factors,
        nav=nav_df,
        analysis_mode=FactorAnalysisMode.FULL,
        risk_free_rate=risk_free_rate,
    )
    betas, t_stats, rsq, resid = exposure.run()

    if betas is None or betas.empty:
        return pd.DataFrame()

    last_betas = betas.iloc[-1].drop(labels=["const"], errors="ignore")
    last_t = t_stats.iloc[-1].drop(labels=["const"], errors="ignore")

    # Use the Factor Lens definitions for user-facing names and
    # residualization parents so that the tooltip can briefly
    # explain what sits behind each factor. The betas dataframe
    # is indexed by factor *names* (e.g. "Equity", "Rates"), so
    # we key this mapping by name rather than by ETF ticker.
    factor_defs = {f.name: f for f in FACTOR_LENS_UNIVERSE}

    factor_labels = []
    descriptions = []
    residualizations = []
    for tick in last_betas.index:
        fdef = factor_defs.get(tick)
        if fdef is not None:
            factor_labels.append(fdef.name)

            # Definition for the tooltip, spelling out
            # the proxy instrument and whether the factor is residualized.
            base_desc = fdef.description or "factor proxy ETF"
            if fdef.parents:
                descriptions.append(
                    f"{fdef.name} factor: proxied by {fdef.ticker} "
                    f"({base_desc}); returns are residualized vs "
                    + ", ".join(fdef.parents)
                    + "."
                )
                residualizations.append(
                    "Child factor: residualized vs "
                    + ", ".join(fdef.parents)
                    + " to remove overlapping risk."
                )
            else:
                descriptions.append(
                    f"{fdef.name} factor: proxied by {fdef.ticker} "
                    f"({base_desc}); top-level, not residualized."
                )
                residualizations.append(
                    "Top-level factor (not residualized)."
                )
        else:
            factor_labels.append(tick)
            descriptions.append(
                "See methodology section for factor definition and construction details."
            )
            residualizations.append(
                "Residualization details: see methodology section."
            )

    df = pd.DataFrame({
        "factor": factor_labels,
        "beta": last_betas.values,
        "t_stat": last_t.reindex(last_betas.index).values,
        "description": descriptions,
        "residualization": residualizations,
    })

    df["significant"] = df["t_stat"].abs() >= 1.96
    df["sign_bucket"] = np.where(df["beta"] >= 0, "pos", "neg")
    df["sig_bucket"] = np.where(df["significant"], "sig", "nonsig")
    df["color_key"] = df["sign_bucket"] + "_" + df["sig_bucket"]

    df["significance_note"] = (
        "Bar color encodes sign and significance of beta "
        "(darker = statistically significant)."
    )

    return df


def render_factor_tab(
    base_currency: str,
    start_date,
    end_date,
    rebal_policy_name: str,
    w_equity: float,
    portfolio_name: str,
):
    st.title(f"Factor Exposures – {portfolio_name}")
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

    # Read-only summary of the configuration coming from the main
    # controls (Tab 1). Users cannot change these here to keep
    # both tabs perfectly in sync.
    st.caption(
        f"Portfolio Configuration: {base_currency} | {start_date} → {end_date} | "
        f"Rebal: {rebal_label} | Equity weight: {w_equity:.0%} | "
        f"Costs: {tc_bps:.1f} bps per trade"
    )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Running factor exposure analysis..."):
        ctx = build_analysis_context(
            base_currency=base_currency,
            start_date=start_date,
            end_date=end_date,
            rebal_policy_name=rebal_policy_name,
            w_equity=w_equity,
            portfolio_name=portfolio_name,
        )

    if ctx is None:
        st.error("Could not build analysis context for the selected range.")
        return

    bt = ctx["bt"]
    rf = ctx["rf"]
    factors_prices = ctx["factors_prices"]

    nav = bt.nav.dropna()

    # Align factors to NAV: exposure is computed on the overlapping
    # window only, while the factor indices themselves are built
    # from the full history in the backend helper.
    common_idx = nav.index.intersection(factors_prices.index)
    if common_idx.empty:
        st.caption("No overlapping history between NAV and factors.")
        return

    risk_factors_aligned = factors_prices.loc[common_idx]
    nav_aligned = nav.loc[common_idx]

    # Top row: latest betas (left) and rolling betas (right)
    top_left, top_right = st.columns([1, 1])

    with top_left:
        beta_table = _build_latest_beta_table(
            risk_factors=risk_factors_aligned,
            nav=nav_aligned,
            risk_free_rate=rf,
        )

        if beta_table.empty:
            st.caption(
                "No factor exposures available for the selected configuration.")
        else:
            # Use the last common date as the "as of" timestamp
            as_of_date = common_idx[-1].date()
            st.subheader(f"Latest factor betas (as of {as_of_date})")

            color_scale = alt.Scale(
                domain=["pos_sig", "pos_nonsig", "neg_sig", "neg_nonsig"],
                range=RANGE_COLORS,
            )

            chart = (
                alt.Chart(beta_table)
                .mark_bar()
                .encode(
                    y=alt.Y("factor:N", sort="-x", title=None),
                    x=alt.X("beta:Q", title="Beta"),
                    color=alt.Color(
                        "color_key:N", scale=color_scale, legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("factor:N", title="Factor"),
                        alt.Tooltip("description:N", title="Definition"),
                        alt.Tooltip(
                            "residualization:N", title="Residualization"
                        ),
                        alt.Tooltip(
                            "significance_note:N",
                            title="Color encoding",
                        ),
                        alt.Tooltip(
                            "beta:Q", title="Beta", format=".2f"
                        ),
                        alt.Tooltip(
                            "t_stat:Q", title="t-stat", format=".2f"
                        ),
                    ],
                )
                .properties(height=420)
            )

            st.altair_chart(chart, width='stretch')

    # Rolling factor betas over time (reusing integration script parameters)
    exposure_roll = FactorExposure(
        risk_factors=risk_factors_aligned,
        nav=nav_aligned.to_frame("NAV"),
        analysis_mode=FactorAnalysisMode.ROLLING,
        lookback=120,
        smoothing_window=5,
        risk_free_rate=rf,
    )
    betas_roll, t_stats_roll, rsq_roll, resid_roll = exposure_roll.run()

    if betas_roll is None or betas_roll.empty:
        st.caption(
            "No rolling factor betas available for the selected configuration.")
        return

    betas_factors = betas_roll.drop(columns=["const"], errors="ignore")
    t_stats_factors = t_stats_roll.drop(columns=["const"], errors="ignore")

    with top_right:
        st.subheader("Rolling factor betas (6M window)")

        betas_long = betas_factors.reset_index().melt(
            id_vars=betas_factors.index.name or "index",
            var_name="factor",
            value_name="beta",
        )
        betas_long = betas_long.rename(
            columns={betas_factors.index.name or "index": "date"})

        t_long = t_stats_factors.reset_index().melt(
            id_vars=t_stats_factors.index.name or "index",
            var_name="factor",
            value_name="t_stat",
        )
        t_long = t_long.rename(
            columns={t_stats_factors.index.name or "index": "date"})

        betas_long = betas_long.merge(
            t_long, on=["date", "factor"], how="left")

        # Drop rows where beta is NaN so the chart starts at the
        # first point with data instead of showing an empty chunk.
        betas_long = betas_long.dropna(subset=["beta"])

        # Consistent colours for factor lines using the shared mapping
        f_unique = betas_long["factor"].astype(str).unique().tolist()
        f_domain = [name for name in FACTOR_COLOR_DOMAIN if name in f_unique]
        if f_domain:
            f_range = [FACTOR_COLOR_MAP[name] for name in f_domain]
            f_scale = alt.Scale(domain=f_domain, range=f_range)
        else:
            f_scale = alt.Undefined

        line_chart = (
            alt.Chart(betas_long)
            .mark_line()
            .encode(
                x=alt.X("date:T", axis=alt.Axis(format="%b %Y", title=None)),
                y=alt.Y("beta:Q", title="Beta"),
                color=alt.Color(
                    "factor:N",
                    legend=alt.Legend(title="Factor"),
                    scale=f_scale,
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("factor:N", title="Factor"),
                    alt.Tooltip("beta:Q", title="Beta", format=".3f"),
                    alt.Tooltip("t_stat:Q", title="t-stat", format=".2f"),
                ],
            )
            .properties(height=420)
        )

        st.altair_chart(line_chart, width='stretch')

    # Factor price series (left) and correlation heatmap (right)
    try:
        # For prices, use the full factor history from the context
        factors_for_prices = factors_prices
        if factors_for_prices is None or factors_for_prices.empty:
            st.caption("No factor price history available.")
            return

        # For correlations, we can restrict to the overlapping NAV window
        factors_window = factors_prices.loc[common_idx]

        # Lay out factor price history and correlation matrix side by side
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.subheader("Factor price indices")

            factors_long = (
                factors_for_prices
                .reset_index()
                .melt(
                    id_vars=factors_for_prices.index.name or "index",
                    var_name="factor",
                    value_name="price",
                )
            )
            factors_long = factors_long.rename(
                columns={factors_for_prices.index.name or "index": "date"}
            )

            # Consistent colours for factor price indices using the same
            # mapping as the pies and rolling betas.
            p_unique = factors_long["factor"].astype(str).unique().tolist()
            p_domain = [
                name for name in FACTOR_COLOR_DOMAIN if name in p_unique
            ]
            if p_domain:
                p_range = [FACTOR_COLOR_MAP[name] for name in p_domain]
                p_scale = alt.Scale(domain=p_domain, range=p_range)
            else:
                p_scale = alt.Undefined

            price_chart = (
                alt.Chart(factors_long)
                .mark_line()
                .encode(
                    x=alt.X(
                        "date:T",
                        axis=alt.Axis(format="%b %Y", title=None),
                    ),
                    y=alt.Y("price:Q", title="Index level"),
                    color=alt.Color(
                        "factor:N",
                        legend=alt.Legend(title="Factor"),
                        scale=p_scale,
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("factor:N", title="Factor"),
                        alt.Tooltip("price:Q", title="Index", format=".2f"),
                    ],
                )
                .properties(height=360)
            )

            st.altair_chart(price_chart, width='stretch')

        f_ret = get_returns(
            factors_window,
            lookback=1,
            method=ReturnMethod.LOG,
        ).dropna()

        if f_ret.empty:
            st.caption(
                "Could not compute factor correlation matrix: empty factor returns.")
        else:
            corr = f_ret.corr()
            corr.reset_index(inplace=True)
            corr_long = corr.melt(
                id_vars=corr.columns[0],
                var_name="factor_col",
                value_name="corr",
            )
            corr_long = corr_long.rename(
                columns={corr.columns[0]: "factor_row"})

            with right_col:
                st.subheader("Factor return correlation matrix")

                corr_chart = (
                    alt.Chart(corr_long)
                    .mark_rect()
                    .encode(
                        x=alt.X("factor_col:N", title=None),
                        y=alt.Y("factor_row:N", title=None),
                        color=alt.Color(
                            "corr:Q",
                            scale=alt.Scale(
                                scheme=DIVERGING_SCHEME_BLUE_ORANGE),
                            title="Corr",
                        ),
                        tooltip=[
                            alt.Tooltip("factor_row:N", title="Row"),
                            alt.Tooltip("factor_col:N", title="Col"),
                            alt.Tooltip(
                                "corr:Q", title="Correlation", format=".2f"
                            ),
                        ],
                    )
                    .properties(height=360)
                )

                st.altair_chart(corr_chart, width='stretch')
    except Exception as e:
        st.caption(f"Could not compute factor correlation matrix: {e}")
