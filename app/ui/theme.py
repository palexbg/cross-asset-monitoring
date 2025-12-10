"""Shared UI theme constants for the app.

Place common color maps and other visual constants here so all tabs
reuse the same definitions instead of duplicating them.
"""

# Shared colour mapping for factor-style buckets so that labels like
# "Equity" and "Rates/Bonds" use consistent colours across charts.
FACTOR_COLOR_MAP = {
    "Equity": "#1f77b4",
    "Bonds": "#ff7f0e",
    "Rates": "#ff7f0e",
    "Credit": "#2ca02c",
    "Commodities": "#d62728",
    "ForeignCurrency": "#9467bd",
    "FX": "#9467bd",
    "Momentum": "#e377c2",
    "Value": "#8c564b",
}

FACTOR_COLOR_DOMAIN = list(FACTOR_COLOR_MAP.keys())

# Default color used when a factor name is not known in the map
DEFAULT_MISSING_COLOR = "#cccccc"

# A small palette used in a few charts for green->red ranges
RANGE_COLORS = ["#2ca02c", "#98df8a", "#d62728", "#ff9896"]

# Altair color scheme names used in the app (keep as strings so
# visualization libraries can use them directly).
DIVERGING_SCHEME_BLUE_ORANGE = "blueorange"
HEATMAP_SCHEME_RED_YELLOW_GREEN = "redyellowgreen"

# Highlight color used for area charts (drawdown shading)
AREA_HIGHLIGHT_COLOR = "#d62728"
