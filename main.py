from backend.config import DataConfig

from app.ui.tab1_overview import render_overview_tab
from app.ui.tab2_factors import render_factor_tab
from app.ui.tab3_risk_contrib import render_risk_contrib_tab
from app.ui.tab4_methodology import render_methodology_tab
from app.ui.analysis_context import build_analysis_context
from app.ui.sidebar_agent import render_ai_analyst_sidebar, prepare_context_agent

from pathlib import Path

import streamlit as st
import pandas as pd


# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Cross-Asset Portfolio Monitor",
    layout="wide"
)


# -----------------------
# Sidebar controls (Tab 1 scope)
# -----------------------
st.sidebar.header("Controls")
# Compact data source notice
st.sidebar.caption("Data source: Yahoo Finance")

base_currency = st.sidebar.selectbox(
    "Portfolio currency",
    options=["USD", "EUR", "CHF"],
    index=0
)

# Date inputs: use two explicit controls (start / end)
default_start = pd.Timestamp("2018-01-01").date()
min_end_for_all_factors = pd.Timestamp("2022-06-01").date()


def _latest_date_from_csv(path: str):
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
    return df.index.max().date()


csv_latest = _latest_date_from_csv(DataConfig.etf_data_path)
end_date_default = csv_latest or pd.Timestamp.today().date()
if end_date_default < min_end_for_all_factors:
    end_date_default = min_end_for_all_factors

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
)

end_date = st.sidebar.date_input(
    "End date",
    value=end_date_default,
    min_value=min_end_for_all_factors,
    help=(
        "To include all factors (many start in mid-2022), the end date "
        "must be on or after 2022-06-01. If you need earlier dates, "
        "some factors will be unavailable."
    ),
)

min_history_days = int(3.5 * 365)
min_allowed_start = (pd.Timestamp(end_date) -
                     pd.Timedelta(days=min_history_days)).date()

if start_date > min_allowed_start:
    st.sidebar.warning(
        f"Start date adjusted to {min_allowed_start} to ensure ~3.5y history."
    )
    override = st.sidebar.checkbox(
        "Override start date (may break factors)",
        value=False,
    )
    if not override:
        start_date = min_allowed_start

# COMPACT UI: Move the long text into a collapsed expander to save space
with st.sidebar.expander("ℹ️ Date Constraints Info"):
    st.caption(
        "Note: The end-date restriction exists because Yahoo Finance does not "
        "provide sufficiently long histories for several factor ETFs (many "
        "start in mid-2022). We also require at least 3.5 years between start "
        "and end to produce stable factor exposure series."
    )

portfolio_preset = st.sidebar.selectbox(
    "Portfolio preset",
    options=["60/40 Monthly Rebal", "60/40 Buy&Hold", "Custom Equity/Bond"],
    index=0
)

if portfolio_preset == "60/40 Buy&Hold":
    rebal_policy_name = "One-time only at beginning"
    st.sidebar.caption(
        "Rebalance schedule: One-time only at beginning for 60/40 Buy&Hold.")
else:
    rebal_policy_name = st.sidebar.selectbox(
        "Rebalance schedule",
        options=["US Month Start", "US Month End"],
        index=0,
    )

if portfolio_preset in ["60/40 Monthly Rebal", "60/40 Buy&Hold"]:
    w_equity = 0.60
else:
    w_equity = st.sidebar.slider("Equity weight (SPY)", 0.0, 1.0, 0.60, 0.05)


# Run the orchestrator analysis script to build all shared context
with st.spinner("Analyzing Portfolio..."):
    pf_analytics = build_analysis_context(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        rebal_policy_name=rebal_policy_name,
        w_equity=w_equity,
        portfolio_name=portfolio_preset,
    )

# ---------------------------------------------------------------------
# Create Context for the Agent
# ---------------------------------------------------------------------
agent_context = prepare_context_agent(
    pf_analytics=pf_analytics,
    base_currency=base_currency,
    portfolio_preset=portfolio_preset,
    start_date=start_date,
    end_date=end_date,
)

# -----------------------
# RENDER AI ANALYST SIDEBAR
# -----------------------
render_ai_analyst_sidebar(agent_context)


# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Tab 1 - Overview",
    "Tab 2 - Risk Factor Lens",
    "Tab 3 - Risk & Return Contributions",
    "Tab 4 - Methodology Details",
])

with tab1:
    render_overview_tab(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        rebal_policy_name=rebal_policy_name,
        w_equity=w_equity,
        portfolio_name=portfolio_preset,
    )

with tab2:
    render_factor_tab(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        rebal_policy_name=rebal_policy_name,
        w_equity=w_equity,
        portfolio_name=portfolio_preset,
    )

with tab3:
    render_risk_contrib_tab(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        rebal_policy_name=rebal_policy_name,
        w_equity=w_equity,
        portfolio_name=portfolio_preset,
    )

with tab4:
    render_methodology_tab()
