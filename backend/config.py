"""Configuration dataclasses for data loading, risk, backtest and factors."""

from dataclasses import dataclass
from .structs import FactorDef, ReturnMethod, CovarianceMethod, ComputeOn

# --------------------------------
# OpenAI Config
# ----------------------------------


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the OpenAI Agent (Ned)."""
    model_name: str = "gpt-4o-mini"
    # Determinism Settings
    temperature: float = 0.0         # 0.0 = Maximum consistency
    top_p: float = 1.0               # Standard pairing with Temp 0

    # Constraints
    max_tokens: int = 4000

    # Penalties - 0 for technical definitions
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Timeout (Streamlit UX safety)
    timeout_seconds: int = 15

# ----------------------------------
# ASSET RISK CONFIGURATION
# ----------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Global data-related settings used across the backend."""

    # Path to the cached CSV containing ETF close prices used as a local convenience cache
    etf_data_path: str = 'cached_etf_close_prices.csv'
    maxfill_days: int = 5  # maximum number of consecutive missing days to fill forward


@dataclass(frozen=True)
class FactorRiskConfig:
    """Configuration for factor risk calculations (covariance and returns)."""
    span: int = 63  # days
    returns_method: ReturnMethod = ReturnMethod.SIMPLE
    cov_method: CovarianceMethod = CovarianceMethod.EWMA
    compute_on: ComputeOn = ComputeOn.REBAL_ONLY
    annualization_factor: int = 252
    smoothing_window: int = 5  # days for the factor betas construction


# ----------------------------------
# ASSET RISK CONFIGURATION
# ----------------------------------

@dataclass(frozen=True)
class AssetRiskConfig:
    """Configuration for asset-level risk calculations (covariance and returns)."""
    cov_method: CovarianceMethod = CovarianceMethod.EWMA
    span: int = 63  # days
    returns_method: ReturnMethod = ReturnMethod.SIMPLE
    compute_on: ComputeOn = ComputeOn.REBAL_ONLY
    annualization_factor: int = 252

# ----------------------------------
# BACKTEST CONFIGURATION
# ----------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the generic long-only backtester."""
    initial_cash: float = 1_000_000.0
    cost_rate: float = 0.0010
    trade_at_close: bool = True
    use_last_known_price: bool = True
    reinvest_proceeds: bool = True
    interest_rate: float = 0.0

# ----------------------------------
# FACTOR LENS CONFIGURATION
# Heavily inspired by https://www.venn.twosigma.com/resources/factor-lens-update
# ----------------------------------


@dataclass(frozen=True)
class FactorConfig:
    """Configuration for factor lens construction and scaling behaviour."""
    smoothing_window: int = 5  # days
    lookback_window: int = 120  # days
    cov_span: int = 750  # days
    target_yearly_vol: float = 0.15  # 15% annualized volatility
    scale_factors: bool = True

 # Factor Lens Universe Definitions
FACTOR_LENS_UNIVERSE = [
    # Tier 1
    FactorDef("Equity", "VT", description="Vanguard Total World Stock ETF"),
    FactorDef("Rates", "IEF", description="iShares 7-10 Year Treasury Bond ETF"),

    # Tier 2 (Residualized against Tier 1)
    FactorDef("Credit", "LQD", parents=[
              "Equity", "Rates"], description="iShares iBoxx $ Investment Grade Corporate Bond ETF"),
    FactorDef("Commodities", "GSG", parents=[
              "Equity", "Rates"], description="iShares S&P GSCI Commodity-Indexed Trust"),

    FactorDef("ForeignCurrency", "UDN", parents=[
              "Equity", "Rates"], description="(Triangulated) FX Factor G10 vs Base Currency"),

    # Tier 3 (Residualized against Tier 1 & 2)
    FactorDef("Value", "VTV", description="Vanguard Value Index Fund ETF Shares", parents=[
              "Equity", "Rates", "Credit", "Commodities"]),
    FactorDef("Momentum", "MTUM", description="iShares MSCI USA Momentum Factor ETF", parents=[
              "Equity", "Rates", "Credit", "Commodities"]),
]
