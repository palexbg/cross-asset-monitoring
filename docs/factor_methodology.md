# Methodology — Cross-Asset Factor Lens

This dashboard implements an **educational demonstration** of a compact, transparent cross-asset “factor lens” workflow using investable ETF proxies and a small parent–child factor hierarchy. The design is motivated by the following public reference on incorporating historical portfolio and factor analysis into monitoring workflows, adapted from 
[Incorporating Historical Portfolio Analysis into your Workflows](https://www.venn.twosigma.com/resources/incorporating-historical-portfolio-analysis-into-your-workflows)

 into a local, auditable implementation with public data. 
The objective is to provide an interpretable monitoring layer that connects:
1) portfolio performance and allocation drift,  
2) factor construction with hierarchical orthogonalization,  
3) exposure estimation, and  
4) asset and factor risk/return contributions.

The scope is intentionally narrow to keep the lens readable and to emphasize methodology and engineering structure over breadth of instrument coverage.

---

## Factor universe

The factor set is intentionally small and cross-asset, expressed through liquid ETF proxies that were hand-picked. The tickers here are from Yahoo Finance. The generality of the methodology is not tied to these specific proxies; they serve as convenient public data proxies for the demonstration:

- **Equity**: `VT`  
- **Rates**: `IEF`  
- **Credit**: `LQD`  
- **Commodities**: `GSG`  
- **Foreign Currency**: `UDN` (triangulated when base currency is not USD)  
- **Value**: `VTV`  
- **Momentum**: `MTUM`

These serve as investable, public proxies for broad macro and style drivers.

---

## Core design choice: hierarchical orthogonalization of the (macro) risk factor set

A central methodological choice is **parent–child residualization (orthogonalization)**. Rather than treating factor proxies as independent, several “child” factors are orthogonalized against economically intuitive “parent” factors. This reduces double-counting and improves interpretability of exposures and risk contributions.

### Parent–child map

- **Credit** is residualized on **Equity** and **Rates**.  
- **Commodities** is residualized on **Equity** and **Rates**.  
- **Foreign Currency** is residualized on **Equity** and **Rates**.  
- **Value** is residualized on **Equity**, **Rates**, **Credit**, and **Commodities**.  
- **Momentum** is residualized on **Equity**, **Rates**, **Credit**, and **Commodities**.

The result is a compact orthogonalized set where child factors represent more incremental drivers rather than repackaged broad market exposure.

---

## Factor construction (high-level)

All analytics are performed in the selected **base currency** (USD/EUR/CHF). Prices are normalized into the base currency and aligned with a risk-free series.

The construction pipeline is:

1) Compute 1-day and 5-day **log returns** for all factor proxy ETFs.  
2) Convert the risk-free series to log form and compute **excess returns**.  
3) Estimate a time-varying **EWMA covariance** of 5-day excess returns.  
4) For each child factor, estimate time-varying parent sensitivities using covariance blocks and compute **residualized returns** by removing the parent component.  
5) Optionally scale residualized child factors to stabilize interpretability across time.  
6) Add the risk-free back to express outputs as total-return **factor indices** suitable for plotting and regressions.

This approach keeps the factor lens stable and interpretable with a limited proxy set.

### Performance and scalability

The computationally heavy kernels for EWMA moments and residualization are Numba-accelerated. 

---

## Exposure estimation

Factor exposures are estimated by regressing **portfolio excess returns** on **factor excess returns**.

To reduce noise from asynchronous market closes and day-to-day microstructure effects, regressions are performed on smoothed multi-day returns (with a 5-day return backbone). Two exposure views are supported:

- **Latest betas**: a full-sample regression view of current average exposure.  
- **Rolling betas**: time-varying exposure estimates using a fixed lookback window.

These estimates supply both the monitoring visuals and the inputs for factor risk decomposition.

---

## Risk and return attribution

The attribution layer links the factor lens to portfolio-level monitoring via two complementary views:

### Return contributions

Using estimated betas and factor returns, portfolio returns can be decomposed into:
- contributions attributable to each factor, and  
- a residual component.

A rolling contribution view highlights whether recent performance is dominated by a narrow set of drivers, whether the driver mix is stable, and whether factor effects are expanding or fading through time.

### Risk contributions

Portfolio volatility is decomposed into:
- **asset** contributions,  
- **asset-class** contributions, and  
- **factor** contributions.

Factor risk contributions are computed by combining:
1) orthogonalized factor covariance, and  
2) estimated portfolio factor exposures.

This enables a practical split between:
- **systematic (factor-driven)** risk, and  
- **idiosyncratic (residual)** risk.

The orthogonalized hierarchy is essential here: it helps ensure that factor risk contributions reflect distinct drivers rather than overlapping macro exposures.

---

## Tab guide

### Tab 1 — Overview

The portfolio-first monitoring view. It provides:
- NAV and performance summary,  
- drawdowns and rolling volatility,  
- turnover/transaction cost context, and  
- weight drift vs target allocations, to verify that the implementation stayed close to intent.

### Tab 2 — Risk Factor Lens

The factor-centric monitoring view. It shows:
- constructed factor indices,  
- latest factor betas with statistical context,  
- rolling betas, and  
- factor correlation checks.

This tab answers the question of what is the portfolio exposed to, and how stable are those exposures through time.

### Tab 3 — Risk & Return Contributions

It combines:
- portfolio weights and backtest history,  
- factor construction outputs, and  
- exposure estimates

to display:
- rolling factor contribution trends, and  
- asset/asset-class/factor risk contribution pies and tables,  
- including systematic vs idiosyncratic breakdowns.

This tab sheds light on which factor drivers explain recent returns and current volatility.

### Tab 4 — Methodology

Renders this document to keep assumptions, factor definitions, and attribution logic auditable within the app.

---

## Data source and limitations

- The prototype uses public ETF proxies and pricing sourced through a Yahoo Finance-based loader.  
- Some factor proxies have limited history before 2022; date guardrails exist to preserve interpretability of rolling exposure and attribution views.  
- The factor set is intentionally compact and investable, not a broad academic factor zoo.  
- With broader institutional datasets, the same architecture could support a richer factor taxonomy, longer histories, and more granular monitoring.

---

## Disclaimer

This dashboard and codebase are provided for educational and demonstration purposes only.  
They do not constitute investment research, trading advice, or a recommendation to buy or sell any security.  
No guarantee is provided regarding accuracy, completeness, or suitability for any purpose.
