from backend.config import DataConfig
from pathlib import Path
import streamlit as st
import pandas as pd

from app.ui.tab1_overview import render_overview_tab
from app.ui.tab2_factors import render_factor_tab
from app.ui.tab3_risk_contrib import render_risk_contrib_tab
from app.ui.tab4_methodology import render_methodology_tab

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

# Data source notice under the Controls section
st.sidebar.write("Data source: Yahoo Finance")

base_currency = st.sidebar.selectbox(
    "Portfolio currency",
    options=["USD", "EUR", "CHF"],
    index=0
)

# Date inputs: use two explicit controls (start / end) rather than a range
# picker which showed a non-working dropdown in some browsers.
default_start = pd.Timestamp("2018-01-01").date()

# To ensure all factors have sufficient history (many start in mid-2022),
# require the end date to be on or after `min_end_for_all_factors` and
# show a small tooltip explaining this.
min_end_for_all_factors = pd.Timestamp("2022-06-01").date()

# Derive a sensible default end-date. Prefer the latest date available in the
# local cached CSV (if present) so the demo uses the freshest available data;
# otherwise fall back to today's date. Always ensure the default is at least
# `min_end_for_all_factors` so the UI doesn't present an obviously unusable
# default.


def _latest_date_from_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
        if df.index.size:
            return df.index.max().date()
    except Exception:
        # If anything goes wrong reading the cache, ignore and fallback
        return None


csv_latest = _latest_date_from_csv(DataConfig.etf_data_path)
end_date_default = csv_latest or pd.Timestamp.today().date()
# Ensure default is not earlier than the minimum required end date
if end_date_default < min_end_for_all_factors:
    end_date_default = min_end_for_all_factors

# Start date first (UI preference)
start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
)

# End date; enforce a minimum end date so factor histories are available.
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

# Require at least 3.5 years (~1278 days) of history between start and end
# so that factor exposure series (rolling regressions, lookbacks) are stable.
min_history_days = int(3.5 * 365)
min_allowed_start = (pd.Timestamp(end_date) -
                     pd.Timedelta(days=min_history_days)).date()

# If the selected start is too recent relative to end, auto-adjust (with override)
if start_date > min_allowed_start:
    st.sidebar.warning(
        f"Start date adjusted to {min_allowed_start} to ensure at least 3.5 years "
        f"of history for factor exposure calculations (you selected {start_date})."
    )
    override = st.sidebar.checkbox(
        "Override and use selected start date (may produce incomplete factor exposures)",
        value=False,
    )
    if not override:
        start_date = min_allowed_start

# Explain why the end date is restricted and why the start must be earlier
st.sidebar.caption(
    "Note: the end-date restriction exists because Yahoo Finance does not "
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
    # Buy&Hold invests once at inception, so the calendar schedule
    # is irrelevant. Show a greyed-out note instead of an active
    # control.
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
