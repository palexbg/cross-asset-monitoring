import streamlit as st
import pandas as pd

from app.ui.tab1_overview import render_overview_tab

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

base_currency = st.sidebar.selectbox(
    "Portfolio currency",
    options=["USD", "EUR", "CHF"],
    index=1
)

# Date range
end_date_default = pd.Timestamp("2025-11-28").date()
default_start = pd.Timestamp("2018-01-01").date()
date_input = st.sidebar.date_input(
    "Date range",
    value=(default_start, end_date_default)
)

# Streamlit can return a single date or a tuple during interaction
if isinstance(date_input, tuple):
    if len(date_input) == 2:
        start_date, end_date = date_input
    elif len(date_input) == 1:
        # Treat a single picked date as both start and end for now
        start_date = end_date = date_input[0]
    else:
        start_date, end_date = default_start, end_date_default
else:
    # Single date selected (e.g. during range selection)
    start_date = end_date = date_input

rebal_policy_name = st.sidebar.selectbox(
    "Rebalance schedule",
    options=["US Month Start", "US Month End"],
    index=0
)

portfolio_preset = st.sidebar.selectbox(
    "Portfolio preset",
    options=["60/40 SAA", "Custom SAA"],
    index=0
)

if portfolio_preset == "60/40 SAA":
    w_equity = 0.60
else:
    w_equity = st.sidebar.slider("Equity weight (SPY)", 0.0, 1.0, 0.60, 0.05)

# how_qs_report = st.sidebar.checkbox("Show QuantStats HTML report", value=True)


# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs([
    "Tab 1 - Overview",
    "Tab 2 - Factor lens",
    "Tab 3 - Risk"
])

with tab1:
    render_overview_tab(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        rebal_policy_name=rebal_policy_name,
        w_equity=w_equity
    )

with tab2:
    st.info("Factor lens tab coming next. We'll wire FactorConstruction + FactorExposure here.")

with tab3:
    st.info("Risk tab coming next. We'll wire AssetRiskEngine + FactorRiskEngine here.")
