"""
Methodology Definitions and Explanations for the UI
"""
import textwrap
import inspect

import backend.config as cfg
from backend.universes import (
    get_investment_universe,
    get_factor_universe,
    get_fx_universe
)

# =========================================================================
# STATIC DEFINITIONS of methodology
# =========================================================================
STATIC_DEFINITIONS = {
    "Goal of the Cross-asset monitoring tool with a cross-asset factor-based lens": """
This dashboard implements an **educational demonstration** of a compact, transparent cross-asset "factor-based lens" workflow using investable ETF proxies and a small parent–child factor hierarchy.

The design draws inspiration from industry research on cross-asset factor-based risk analysis (e.g., publicly available e.g. [here](https://www.venn.twosigma.com/resources/incorporating-historical-portfolio-analysis-into-your-workflows)).

The objective is to provide an interpretable monitoring layer that connects:
1) portfolio performance and allocation drift,
2) factor construction with hierarchical orthogonalization,
3) exposure estimation, and
4) asset and factor risk/return contributions.
""",

    "Factor Hierarchy Structure": """
The orthogonalization order forms a dependency tree:

1. ROOTS (Independent):
   - Equity (VT)
   - Rates (IEF)

2. LEVEL 1 (Residualized on Roots):
   - Credit (LQD) -> depends on [Equity, Rates]
   - Commodities (GSG) -> depends on [Equity, Rates]
   - FX (UDN) -> depends on [Equity, Rates]

3. LEVEL 2 (Residualized on Roots + Level 1):
   - Value (VTV) -> depends on [Equity, Rates, Credit, Commodities]
   - Momentum (MTUM) -> depends on [Equity, Rates, Credit, Commodities]

This structure ensures that 'Momentum' represents pure price action, stripped of all macro and credit beta.
""",
    "How to Read the Dashboard": """
A recommended interpretation flow:

1. **Factor Exposures** – What risks am I exposed to?
2. **Risk Contribution** – Which risks actually dominate volatility?
3. **Systematic vs Idiosyncratic Risk** – How much is explained by this lens?
4. **Return Attribution** – What tended to help or hurt historically?

This order mirrors how institutional allocators typically reason about
portfolio risk before performance.
""",
    "Risk Factor Orthogonalization": """
We use a **Hierarchical Regression** approach to handle multicollinearity.
A central methodological choice is **parent–child residualization**.

Rather than treating factor proxies as independent, several "child" factors are orthogonalized against economically intuitive "parent" factors.

* **Credit** is residualized on Equity and Rates.
* **Commodities** is residualized on Equity and Rates.
* **Foreign Currency** is residualized on Equity and Rates.
* **Value & Momentum** are residualized on Equity, Rates, Credit, and Commodities.

This ensures that the Credit factor captures pure credit risk and not just repackaged equity beta.
""",

    "Asset Risk vs. Factor Risk": """
These are two different ways of slicing the **same** Total Risk pie.

1. **Asset Risk (Holdings View):**
   - Answers: *"Which ticker is causing my volatility?"*
   - Example: "SPY contributes 12% to volatility."
   - Note: Since SPY is driven by the market, most "Asset Risk" is actually "Systematic."

2. **Factor Risk (Systematic View):**
   - Answers: *"Which economic driver is causing my volatility?"*
   - Example: "Equity Factor contributes 12% to volatility."

**Common Pitfall:**
Do not confuse **Asset Risk** with **Idiosyncratic Risk**.
- **Asset Risk** = Systematic part of asset + Idiosyncratic part of asset.
- **Idiosyncratic Risk** = Only the unexplained residual noise.
""",


    "Return Conventions": """
The backend distinguishes between simple and log returns through a ReturnMethod enum:

* **Simple returns:** $r_t = P_t / P_{t-1} - 1$
    * Used for: Backtesting, NAV calculation, Annualized Volatility.
* **Log returns:** $r_t = ln(P_t / P_{t-1})$
    * Used for: Factor construction, Orthogonalization, Regression (additivity).

In the exposure engine, both portfolio NAV and factors are converted into smoothed 5-day log excess returns before running the regressions.
""",

    "Lookback Window": """
We require roughly 3.5 calendar years of history between the selected
start and end dates. The UI enforces this by auto-adjusting the start date
when needed so that rolling factor exposure regressions have enough data
to be stable.
""",
    "Definition: NAV": """
    Net Asset Value (NAV) represents the per-share value of the portfolio.
    In this dashboard, it tracks the cumulative performance of the strategy,
    indexed to start at 1.0 (or $1) at inception.
    """,
    "Risk-Free Rate Construction": """
Risk-free returns are constructed dynamically based on the chosen base currency:

* **USD:** ^IRX (13-week T-bill yield index).
* **EUR:** EL4W.DE (EUR money-market ETF).
* **CHF:** CSBGC3.SW (CHF short-term government bond ETF).


The resulting daily series is used consistently in factor construction, exposure estimation, and performance analytics.
""",

    "Annualization and Performance Metrics": """
Performance statistics follow standard daily-to-annual conversions:

* **Historical Volatility:** std of realized daily simple returns × $\sqrt{252}$.
* **Sharpe ratio:** mean of daily excess returns / std × $\sqrt{252}$.
* **CAGR:** Geometric mean of the NAV path.

Whenever a risk-free series is available, metrics are calculated on excess returns.
""",

    "Methodology Risk Factor Construction": """
All analytics are performed in the selected **base currency** (USD/EUR/CHF).
The construction pipeline is:

1.  Compute 1-day and 5-day **log returns** for all factor proxy ETFs.
2.  Convert the risk-free series to log form and compute **excess returns**.
3.  Estimate a time-varying **EWMA covariance** of 5-day excess returns over
    a {cov_span}-day span.
4.  **Orthogonalize:** Remove 'Parent' beta from 'Child' factors.
5.  **Scale:** Rescale residual factors to target {target_vol_pct} annualized volatility.
6.  **Re-Index:** Convert back to a Price Index for plotting.
""",

    "Exposure Estimation": """
Factor exposures are estimated by regressing **portfolio excess returns** on **factor excess returns**.

* **Smoothing:** We use a 5-day overlapping return window to reduce noise from asynchronous market closes.
* **Modes:** We support 'Latest Beta' (full sample) and 'Rolling Beta' using
  a fixed lookback window (configured in the factor exposure engine).
""",

    "FX Triangulation": """
Portfolio assets are always converted to the chosen base currency (USD, EUR, CHF)
so that NAV is expressed in the investor’s wealth currency.

- For each non-base asset, we look for a direct FX pair (Local/Base, e.g. EURUSD=X)
  or an inverse pair (Base/Local, e.g. USDEUR=X) and convert prices accordingly.
- Separately, we construct a dedicated FX factor from the UDN ETF (G10 vs USD).
  When the base currency is not USD, we triangulate via the <Base>USD=X pair so
  that the factor reflects “foreign currency vs base” rather than pure USD moves.
- Important: Portfolio assets are fully converted into the base currency.
  Factor proxy ETFs remain in their native currency (USD in this prototype),
  while the dedicated FX factor is triangulated to the base currency.
  Currency movements thus show up primarily in the Foreign Currency factor.

This keeps P&L in base currency while isolating currency risk as an explicit factor.
""",

    "Risk Model (EWMA)": """
Volatility and covariance are estimated with an Exponentially Weighted
Moving Average (EWMA) kernel implemented in `compute_ewma_covar`.

- Asset risk span: {asset_span} trading days.
- Factor risk span: {factor_span} trading days.
- Factor construction span: {factor_cov_span} trading days.

The decay parameter is derived from these spans inside the EWMA kernel
(not hard-coded), so “faster” or “slower” memory is controlled by the
span fields in `AssetRiskConfig`, `FactorRiskConfig` and `FactorConfig`.
""",
    "How to Interpret the cross-asset factor-based lens": """
The cross-asset factor-based lens is a **measurement framework**, not a statement of economic truth.

All exposures, risk contributions, and residuals are defined **relative to the chosen factor set, proxy instruments, and orthogonalization order**.

Key implications:
- A factor exposure answers: *“What combination of these factors best explains the observed returns?”*
- It does **not** imply causality or permanence.
- Residual (idiosyncratic) risk is simply what this lens cannot explain — it is **not automatically alpha**.

Changing the factor universe, proxy ETFs, or residualization order would change the results.
""",
    "Orthogonalization Order and Economic Meaning": """
Orthogonalization is a **design choice**, not a discovery.

In this system, factors are residualized in a parent–child hierarchy to align with
economic intuition (e.g., Credit depends on Equity and Rates).

This implies:
- Child factors represent **incremental risk beyond their parents**
- Later factors (e.g., Momentum) are increasingly “pure” but also more model-dependent
- Reordering the hierarchy would change factor meanings and betas

There is no single correct order — this hierarchy reflects a transparent,
interpretable compromise rather than a universal truth.
""",

    "What is meant by factor risk": """
Throughout the dashboard, “factor risk” refers to **modeled volatility**, not total uncertainty.

Specifically:
- Risk is estimated using historical returns and EWMA covariance models
- Tail events, regime shifts, and non-linear losses are not fully captured
- Risk estimates are conditional on the chosen lookback window and decay parameters

The cross-asset factor-based risk lens is most useful for **relative comparisons**, scenario reasoning,
and understanding diversification — not for predicting extreme outcomes.
""",
    "Currency and Hedging Scope": """
All portfolio analytics are expressed in the selected base currency via FX conversion.
However, factor proxies are simple ETF-based stand-ins and are not constructed using a full currency-hedged index methodology.
As a result, some currency effects may appear in multiple places (asset returns, factor proxies, and the explicit FX factor).
""",

    "Risk Decomposition (MCTR)": """
We compute Marginal Contribution to Risk (MCTR) using an Euler decomposition so that contributions add up to total modeled volatility.

Interpretation:
- Risk contribution is not the same as capital weight.
- A small allocation can dominate volatility if it is highly volatile or strongly correlated with the rest of the portfolio.
- Correlation structure is often the real driver of “hidden concentration.”
""",

    "Factor Construction Volatility Scaling": """
After orthogonalization, residual factors are scaled to target {target_vol_pct}
annualized volatility. This normalizes the 'cross-asset factor-based lens' so that a Beta of 1.0
implies a broadly comparable risk contribution, regardless of whether it is
Equity or Momentum.
""",

    "Return Attribution (cross-asset factor-based lens)": """
Factor-based return attribution decomposes portfolio returns into contributions from each risk factor plus a residual component.

$Residual_t = PortfolioExcess_t - \sum (Beta_f \times FactorReturn_f)$

This output powers the "What drove performance?" heatmap.

**Important interpretation note**

Return attribution is inherently noisier than risk attribution.
Small changes in estimated betas or factor returns can materially
change attributed performance over short horizons.

As a result:
- Risk contributions are generally more stable than return contributions
- Return attribution should be interpreted qualitatively, not precisely
- The residual return captures both skill and uncompensated noise

""",

    "Rebalancing Logic": """
* **Logic:** Trade-at-close using the latest available prices.
* **Frequency:** Weights are reset to Strategic Asset Allocation (SAA)
  targets on pre-defined monthly schedules (e.g. first or last US
  business day of the month via RebalPolicies).
* **Drift:** Between rebalances, weights drift based on asset performance
  (buy-and-hold behaviour).
""",

    "Transaction Costs": """
Transaction costs are modeled as a proportional fee on traded notional.

- Default rate: {cost_bps:.0f} bps ({cost_rate:.4f}) per dollar traded (buys + sells).
- At each rebalance, we compute total traded notional required to move from
  current holdings to target weights and deduct cost_rate × traded_notional
  from portfolio cash.

There is no separate modeling of bid/ask spread, market impact, or ticket fees
in this prototype.
""",

    "Data Cleaning": """
* **Forward Fill:** Missing prices are forward-filled up to 5 days
  to bridge holidays and short gaps without trading on stale data.
* **Outliers/NaNs:** Non-finite values are cleaned before computing
  returns and risk metrics.
""",

    "Plot: Cumulative Returns": """
**Tab 1 (Overview)**
Shows the growth of $1 invested at inception.
* **Interpretation:** Compare the slope of the Portfolio (Blue) vs Benchmark (Grey). Steeper slope = Outperformance.
""",

    "Plot: Underwater Chart": """
**Tab 1 (Overview)**
Measures the percentage decline from the previous All-Time High.
* **Interpretation:** Visualizes 'Pain'. Deep valleys are crashes. The width of the valley shows the 'Time to Recovery'.
""",

    "Estimation Uncertainty and Stability": """
Exposure estimates come from historical regressions and can be noisy.

Practical interpretation tips:
- Focus on **large and persistent** exposures (stable across rolling windows).
- Treat small exposures as “maybe” unless they are consistently present.
- Short windows and choppy markets can cause betas to move quickly even when the underlying strategy hasn’t changed.

This is why the dashboard emphasizes exposures + risk contribution + stability over time, rather than a single point estimate.
""",

    "Plot: Factor Exposures": """
**Tab 2 (Factors)**
A bar chart showing regression betas to the factor return series used in this dashboard.

* **Interpretation:** A larger positive beta means the portfolio tends to move with that factor; a negative beta indicates hedge-like behavior.
* Betas here are model-relative: they depend on the factor proxies, orthogonalization order, and (where applicable) volatility scaling.
* Treat betas as a **co-movement diagnostic**, not a causal statement or a precise one-day sensitivity.
""",

    "Plot: Risk Contribution": """
**Tab 3 (Risk)**
A Pie/Bar chart breaking down total volatility.
* **Interpretation:** Shows which asset brings the highest risk to the portfolio. If 'Equity' is the largest slice, your portfolio is driven by the stock market, regardless of your bond holdings.
""",

    "Plot: Rolling Volatility": """
**Tab 1 (Overview)**
Shows the annualized standard deviation over a 6-month window.
* **Interpretation:** Measures the 'Temperature' of the portfolio. Spikes indicate stress or instability.
""",

    "Systematic vs Idiosyncratic Risk": """
The factor risk model decomposes portfolio volatility into two pieces:

- **Systematic risk**: volatility explained by the factor model
  (the cross-asset factor-based lens).
- **Idiosyncratic risk**: residual volatility not explained by
  the factors, i.e. specific to the portfolio implementation.

**Systematic risk (factor-driven)**

1. We take the latest estimated factor betas (B) from the
   FactorExposure engine. These are sensitivities to the
   orthogonalized factor indices (Equity, Rates, Credit, etc.).
2. We build an EWMA covariance tensor of factor daily simple
   returns using {factor_span} days to for an EWMA vol estimator.
3. For each evaluation date t, we compute systematic variance as

       sys_var_t = B_t' Σ_t B_t

   where Σ_t is the factor covariance matrix for date t.
4. Systematic volatility is the square root of that variance,
   optionally annualized by multiplying by sqrt(252).

Intuition: this is the volatility you would expect if only the
modeled factors moved and the residual were zero.

**Idiosyncratic risk (residual)**

1. In the FactorExposure regressions, we fit smoothed 5-day
   portfolio excess returns on smoothed 5-day factor excess
   returns with HAC-robust errors.
2. The regression residual variance (per date or for the
   latest window) is stored as ``IdiosyncraticRisk``.
3. Idiosyncratic volatility is the square root of this residual
   variance, rescaled to an annualized 1-day equivalent using

       idio_vol_annualized ≈ sqrt(resid_var) × sqrt(252 / 5)

   because the regression backbone uses 5-day smoothed returns.

Intuition: this is the volatility of the part of the portfolio
return that the factor set cannot explain — stock picking,
residual sector bets, implementation noise, model misspecification,
and so on.

**UI**

In the Risk & Return Contributions tab, the factor risk section
shows:

    Systematic volatility: X%
    Idiosyncratic volatility: Y%

where X and Y are annualized estimates from the latest available
snapshot.
""",

    "Data Sources and Limitations": """
**Data source**

- All prices are loaded from a local CSV that was originally
  sourced via a Yahoo Finance-based loader.
- Factor proxies are liquid, investable ETFs (e.g. VT, IEF,
  LQD, GSG, UDN, VTV, MTUM) used as public stand-ins for
  broader macro and style factors.
- Risk-free returns use:
    - ^IRX for USD (13-week T-bill yield index),
    - EL4W.DE for EUR (EUR money-market ETF),
    - CSBGC3.SW for CHF (short-term Swiss government bond ETF),
  converted into daily simple returns and cleaned.

**Scope and limitations**

- The factor set is intentionally compact and investable; this
  is a didactic prototype, not a full academic factor zoo.
- Coverage is constrained by ETF histories and Yahoo Finance
  data quality. Some series have limited pre-2020 history.
- Date guardrails in the UI ensure there is enough history
  (e.g. ~3.5 years) for rolling regressions and EWMA risk
  estimates to remain interpretable.
- FX and risk-free proxies are approximations; they are good
  enough for educational monitoring but not for production
  risk systems.

**Intended use**

The entire dashboard is for educational and demonstrational
purposes only. It is not investment advice, and the outputs
should not be used as the sole basis for real-world portfolio
decisions.
""",

    "Factor Types and Practical Expectations": """
In this dashboard we group risks into:
- **Broad market drivers (macro-like)**: typically higher-capacity and more persistent through time.
- **Cross-sectional tilts (style-like)**: often noisier, more regime-dependent, and more sensitive to proxy choice.
- **Residual**: whatever the model cannot explain; this can include manager skill, implementation details, or simply missing factors.

A useful rule of thumb:
macro-style exposures tend to be more stable; style-like and residual components tend to move around more and should be interpreted with extra caution.
""",
    "Proxy Choices and What They Imply": """
All factors here are represented using investable ETF proxies.
This makes the system transparent and reproducible, but it also means:

- Factor behavior reflects the ETF's implementation (index rules, sector tilts, fees, rebalancing).
- Style proxies (e.g., Value/Momentum ETFs) are not pure theoretical portfolios; they can embed broad market exposure and other tilts.
- Orthogonalization reduces overlap with parent factors, but it does not turn a long-only proxy into a perfectly isolated “pure factor.”

If you swap proxies, you will generally change exposures, risk contributions, and residuals.
""",
    "Relationship to External Research": """
This dashboard is an independent, educational implementation inspired by
publicly available academic and industry research on factor-based portfolio analysis.

It is not affiliated with, endorsed by, or a replica of any proprietary platform.

All factor definitions, proxy choices, and modeling decisions are original to
this implementation and chosen for transparency and pedagogical clarity.
"""


}


def _generate_data_docs():
    """Generates descriptions for every asset, factor, and FX instrument in the system."""
    dynamic_docs = {}

    # Process Investment Universe
    for asset in get_investment_universe():
        key = f"Asset: {asset.ticker} ({asset.name}) ({asset.description})"
        desc = (
            f"{asset.name} ({asset.ticker}) is a {asset.asset_class} instrument. "
            f"Description: {asset.description}. Currency: {asset.currency}."
        )
        dynamic_docs[key] = desc

    # Process Factor Universe
    for factor in get_factor_universe():
        key = f"Factor: {factor.name} ({factor.ticker}) ({factor.description})"
        desc = (
            f"The '{factor.name}' factor is proxied by {factor.ticker}. "
            f"Description: {factor.description}."
        )
        dynamic_docs[key] = desc
    # Process FX Universe
    for fx in get_fx_universe():
        key = f"FX Pair: {fx.ticker}, ({fx.description})"
        desc = (
            f"The FX pair {fx.ticker} represents the exchange rate between "
            f"{fx.description}."
        )
        dynamic_docs[key] = desc

    return dynamic_docs


def get_rebal_schedule_docs() -> dict:
    """Generate docs for each predefined rebalancing schedule."""
    out = {}
    try:
        from backend.structs import RebalPolicies, RebalanceSchedule

        for attr in dir(RebalPolicies):
            if attr.startswith("_"):
                continue
            sched = getattr(RebalPolicies, attr)
            # Defensive: only include actual schedules
            if isinstance(sched, RebalanceSchedule):
                key = f"Rebalance Schedule: {sched.name}"
                desc = sched.description or ""
                # Optionally add class-level docstring context
                cls_doc = inspect.getdoc(RebalPolicies) or ""
                if cls_doc and desc:
                    full = f"{desc} ({cls_doc})"
                else:
                    full = desc or cls_doc
                out[key] = full
    except Exception:
        return {}
    return out


def get_config_docs() -> dict:
    """Introspect `backend.config` for docstrings of config classes."""
    out = {}
    try:
        import backend.config as cfgmod
        for name, val in vars(cfgmod).items():
            if inspect.isclass(val):
                out[f"Config: {val.__name__}"] = inspect.getdoc(val) or ""
    except Exception:
        pass
    return out


def get_struct_docs() -> dict:
    """Introspect `backend.structs` for simple docstrings of datatypes."""
    out = {}
    try:
        import backend.structs as s_mod
        for name, val in vars(s_mod).items():
            if inspect.isclass(val):
                out[f"Struct: {val.__name__}"] = inspect.getdoc(val) or ""
    except Exception:
        pass
    return out


# =========================================================================
# export
# =========================================================================


def _get_clean_definitions():
    """Merges definitions and injects ALL active config parameters."""

    # 1. Merge the base dictionaries (Skipping raw Config/Struct docs)
    base = {
        **STATIC_DEFINITIONS,
        **_generate_data_docs(),
        **get_rebal_schedule_docs(),
    }

    # - backtest
    if "Rebalancing Logic" in base:
        timing = "Market Close" if cfg.BacktestConfig.trade_at_close else "Market Open"
        reinvest = "Enabled" if cfg.BacktestConfig.reinvest_proceeds else "Disabled"

    # Append context to the existing description
        base["Rebalancing Logic"] += (
            f"\n\n**Current Configuration:**\n"
            f"- Execution Timing: {timing}\n"
            f"- Dividend Reinvestment: {reinvest}"
        )

    # - risk & return conventions
    if "Annualization and Performance Metrics" in base:
        base["Annualization and Performance Metrics"] += (
            f"\n\n(Current Annualization Factor: {cfg.AssetRiskConfig.annualization_factor} days)"
        )

    if "Return Conventions" in base:
        # Convert Enum to readable string (e.g., ReturnMethod.SIMPLE -> "Simple")
        method = cfg.AssetRiskConfig.returns_method.value.title()
        base["Return Conventions"] += f"\n\n(Current Asset Return Method: **{method}**)"

    # cross-asset factor-based lens
    if "Exposure Estimation" in base:
        base["Exposure Estimation"] += (
            f"\n\n**Current Settings:**\n"
            f"- Smoothing Window: {cfg.FactorConfig.smoothing_window} days\n"
            f"- Rolling Lookback: {cfg.FactorConfig.lookback_window} days"
        )

    # factor vola scaling
    if "Factor Volatility Scaling" in base:
        scaling_status = "Enabled" if cfg.FactorConfig.scale_factors else "Disabled"
        # Format the existing placeholder {target_vol_pct} AND add the boolean status
        base["Factor Volatility Scaling"] = base["Factor Volatility Scaling"].format(
            target_vol_pct=f"{cfg.FactorConfig.target_yearly_vol:.0%}"
        ) + f"\n(Scaling is currently **{scaling_status}**)"

    # ewma
    if "Risk Model (EWMA)" in base:
        base["Risk Model (EWMA)"] = base["Risk Model (EWMA)"].format(
            asset_span=cfg.AssetRiskConfig.span,
            factor_span=cfg.FactorRiskConfig.span,
            factor_cov_span=cfg.FactorConfig.cov_span,
        )
    # transaction costs
    if "Transaction Costs" in base:
        cost_rate = cfg.BacktestConfig.cost_rate
        cost_bps = cost_rate * 10000.0
        base["Transaction Costs"] = base["Transaction Costs"].format(
            cost_rate=cost_rate,
            cost_bps=cost_bps,
        )

    # data cleaning
    if "Data Cleaning" in base:
        base["Data Cleaning"] += (
            f"\n(Fill limit: {cfg.DataConfig.maxfill_days} days)"
        )

    # Clean indentation (same as before)
    clean_dict = {}
    for k, v in base.items():
        if v:
            clean_dict[k] = textwrap.dedent(v).strip()

    return clean_dict


# EXPORT THE CLEAN DICTIONARY
METHODOLOGY_DEFINITIONS = _get_clean_definitions()


def get_definition(key: str):
    """Safe lookup for the UI/AI."""
    return METHODOLOGY_DEFINITIONS.get(key)
