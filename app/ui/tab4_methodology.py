import streamlit as st
from backend.ai.knowledge import get_definition, METHODOLOGY_DEFINITIONS


def render_methodology_tab():
    """Render the methodology / info tab.

    This view dynamically constructs the methodology explanation using the
    SSOT definitions in `knowledge.py`. This ensures that any change to
    configuration (e.g., lookback windows, decay spans, tickers) is
    immediately reflected here without manual markdown updates.
    """

    st.title("Methodology – Factor Lens")
    st.markdown("---")
    st.info(
        "**Disclaimer:** For educational & demonstrational purposes only. "
        "Loads a local CSV of public prices sourced from Yahoo Finance; not investment advice."
    )

    # We build the Markdown string dynamically, injecting the exact definitions
    # used by the AI Agent.

    # 1. INTRODUCTION & GOAL
    intro_section = f"""
# Methodology — Cross-Asset Factor Lens

{get_definition("Goal of the Cross-asset monitoring tool with a cross-asset factor-based lens")}

{get_definition("Relationship to External Research")}

---

## Core design choice: Hierarchical Orthogonalization

{get_definition("Risk Factor Orthogonalization")}

### The Hierarchy
{get_definition("Factor Hierarchy Structure")}

{get_definition("Orthogonalization Order and Economic Meaning")}

### Why use ETF Proxies?
{get_definition("Proxy Choices and What They Imply")}
"""

    # 2. CONSTRUCTION
    construction_section = f"""
---

## Factor Construction Pipeline

All analytics are performed in the selected **base currency**.

{get_definition("Methodology Risk Factor Construction")}

### Volatility Scaling
{get_definition("Factor Construction Volatility Scaling")}

### FX Treatment
{get_definition("FX Triangulation")}

### Risk Model Parameters (EWMA)
{get_definition("Risk Model (EWMA)")}

---

## Exposure Estimation

{get_definition("Exposure Estimation")}

{get_definition("Estimation Uncertainty and Stability")}
"""

    # 3. ATTRIBUTION & RISK
    attribution_section = f"""
---

## Risk & Return Attribution

The dashboard links the factor lens to portfolio-level monitoring via two views:

### 1. Return Contributions
{get_definition("Return Attribution (cross-asset factor-based lens)")}

### 2. Risk Contributions (MCTR)
{get_definition("Risk Decomposition (MCTR)")}

### Systematic vs. Idiosyncratic Risk
{get_definition("Systematic vs Idiosyncratic Risk")}

### Asset vs. Factor Risk
{get_definition("Asset Risk vs. Factor Risk")}
"""

    # 4. VISUALIZATION GUIDE
    viz_section = f"""
---

## Dashboard Guide: How to Interpret

{get_definition("How to Read the Dashboard")}

### Tab 1 — Overview
* **Cumulative Returns**: {get_definition("Plot: Cumulative Returns")}
* **Underwater Chart**: {get_definition("Plot: Underwater Chart")}
* **Rolling Volatility**: {get_definition("Plot: Rolling Volatility")}

### Tab 2 — Risk Factor Lens
{get_definition("Plot: Factor Exposures")}

### Tab 3 — Risk & Return Contributions
{get_definition("Plot: Risk Contribution")}

---

## Data & Limitations

{get_definition("Data Sources and Limitations")}

{get_definition("What is meant by factor risk")}
"""

    # 5. DYNAMIC UNIVERSE LIST
    # We loop through the knowledge keys to list the exact factors currently configured.
    factor_keys = [k for k in METHODOLOGY_DEFINITIONS.keys()
                   if k.startswith("Factor:")]

    universe_section = "\n\n## Appendix: Current Factor Universe\n"
    if factor_keys:
        for k in factor_keys:
            # Format: "Factor: Equity (VT) ..." -> "**Equity (VT)**: Description..."
            clean_name = k.replace("Factor: ", "")
            desc = METHODOLOGY_DEFINITIONS[k]
            universe_section += f"- **{clean_name}**\n  {desc}\n"

    # Combine all sections
    full_markdown = (
        intro_section +
        construction_section +
        attribution_section +
        viz_section +
        universe_section
    )

    # Render
    st.markdown(full_markdown)
