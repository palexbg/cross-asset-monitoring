from dataclasses import dataclass
from .structs import FactorDef, ReturnMethod, CovarianceMethod

# ----------------------------------
# ASSET RISK CONFIGURATION
# ----------------------------------


@dataclass(frozen=True)
class FactorRiskConfig:
    span: int = 63  # days
    returns_method: ReturnMethod = ReturnMethod.SIMPLE
    compute_on: str = 'all'
    annualization_factor: int = 252


# ----------------------------------
# ASSET RISK CONFIGURATION
# ----------------------------------

@dataclass(frozen=True)
class AssetRiskConfig:
    cov_method: CovarianceMethod = CovarianceMethod.EWMA
    span: int = 63  # days
    returns_method: ReturnMethod = ReturnMethod.SIMPLE
    compute_on: str = 'all'
    annualization_factor: int = 252

# ----------------------------------
# BACKTEST CONFIGURATION
# ----------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    cost_rate: float = 0.0010
    trade_at_close: bool = True
    use_last_known_price: bool = True
    reinvest_proceeds: bool = True
    interest_rate: float = 0.0

# ----------------------------------
# FACTOR LENS CONFIGURATION
# Following Two Sigma's Factor Lens setup
# ----------------------------------


@dataclass(frozen=True)
class FactorConfig:
    smoothing_window: int = 5  # days
    lookback_window: int = 120  # days
    smoothing_window: int = 5  # days
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
