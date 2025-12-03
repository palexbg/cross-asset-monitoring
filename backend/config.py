from dataclasses import dataclass, field
from typing import List, Optional
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.offsets import CustomBusinessMonthEnd
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import Optional
from structs import RebalanceSchedule

# REBALANCE SCHEDULES


class RebalPolicies:
    # US month start
    US_MONTH_START = RebalanceSchedule(
        name="US_MS",
        description="First business day of month (US Holidays)",
        offset=CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())
    )

    # US month end
    US_MONTH_END = RebalanceSchedule(
        name="US_ME",
        description="Last business day of month (US Holidays)",
        offset=CustomBusinessMonthEnd(calendar=USFederalHolidayCalendar())
    )


# BACKTEST CONFIGURATION

@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    cost_bps: float = 0.0010
    trade_at_close: bool = True
    use_last_known_price: bool = True
    interest_rate: float = 0.0
    reinvest_proceeds: bool = True

# FACTOR LENS CONFIGURATION


@dataclass(frozen=True)
class FactorDef:
    name: str
    ticker: str
    description: Optional[str] = None
    parents: List[str] = field(default_factory=list)
    target_vol: Optional[float] = 0.10


FACTOR_LENS_UNIVERSE = [
    # Tier 1
    FactorDef("Equity", "VT", description="Vanguard Total World Stock ETF"),
    FactorDef("Rates", "IEF", description="iShares 7-10 Year Treasury Bond ETF"),

    # Tier 2 (Residualized against Tier 1)
    FactorDef("Credit", "LQD", parents=[
              "Equity", "Rates"], description="iShares iBoxx $ Investment Grade Corporate Bond ETF"),
    FactorDef("Commodities", "GSG", parents=[
              "Equity", "Rates"], description="iShares S&P GSCI Commodity-Indexed Trust"),

    # Tier 3 (Residualized against Tier 1 & 2)
    FactorDef("Value", "VTV", description="Vanguard Value Index Fund ETF Shares", parents=[
              "Equity", "Rates", "Credit", "Commodities"]),
    FactorDef("Momentum", "MTUM", description="iShares MSCI USA Momentum Factor ETF", parents=[
              "Equity", "Rates", "Credit", "Commodities"]),
]
