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
    Asset_Class: str
    ticker: str
    description: Optional[str] = None

# ----------------------------------
# FACTOR ANALYSIS MODES
# ----------------------------------


class FactorAnalysisMode(Enum):
    ROLLING = "rolling"
    FULL = "full"


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
