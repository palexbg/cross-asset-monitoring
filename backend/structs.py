from dataclasses import dataclass, field
from typing import Callable, Optional
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.offsets import CustomBusinessMonthEnd
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import List, Optional
from enum import Enum

import pandas as pd


@dataclass(frozen=True)
class Asset:
    name: str
    asset_class: str
    ticker: str
    currency: str = "USD"
    description: Optional[str] = None


# ----------------------------------
# ENUMERATIONS
# ----------------------------------


class FactorAnalysisMode(Enum):
    ROLLING = "rolling"
    FULL = "full"


class ReturnMethod(Enum):
    LOG = "log"
    SIMPLE = "simple"


class CovarianceMethod(Enum):
    EWMA = "ewma"
    SAMPLE = "sample"


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    CHF = "CHF"


class ComputeOn(Enum):
    # compute only latest point (when compute_over_time=False)
    LATEST = "latest"
    ALL = "all"            # over-time on all available dates
    REBAL_ONLY = "rebal_only"  # over-time only on rebal dates


# ----------------------------------
# REBALANCE SCHEDULES
# ----------------------------------


@dataclass(frozen=True)
class RebalanceSchedule:
    name: str
    description: str
    offset: Optional[object] = None
    generator_func: Optional[Callable[[
        pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]] = None


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

# ----------------------------------
# Factor Lens Definitions
# ----------------------------------


@dataclass(frozen=True)
class FactorDef:
    name: str
    ticker: str
    description: Optional[str] = None
    parents: List[str] = field(default_factory=list)
