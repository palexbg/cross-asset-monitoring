"""Core data structures and enums shared across the backend."""

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
    """Simple asset description used for universes and metadata."""

    name: str
    asset_class: str
    ticker: str
    currency: str = "USD"
    description: Optional[str] = None


# ----------------------------------
# ENUMERATIONS
# ----------------------------------


class FactorAnalysisMode(Enum):
    """Whether factor regressions are run once or on a rolling window."""

    ROLLING = "rolling"
    FULL = "full"


class ReturnMethod(Enum):
    """Return convention used throughout the backend."""

    LOG = "log"
    SIMPLE = "simple"


class CovarianceMethod(Enum):
    """Covariance estimation method for risk engines."""

    EWMA = "ewma"
    SAMPLE = "sample"


class Currency(Enum):
    """Supported base currencies for the system."""

    USD = "USD"
    EUR = "EUR"
    CHF = "CHF"


class ComputeOn(Enum):
    """Controls on which dates risk calculations are performed."""

    # compute only latest point (when compute_over_time=False)
    LATEST = "latest"
    ALL = "all"            # over-time on all available dates
    REBAL_ONLY = "rebal_only"  # over-time only on rebal dates


# ----------------------------------
# REBALANCE SCHEDULES
# ----------------------------------


@dataclass(frozen=True)
class RebalanceSchedule:
    """Definition of a calendar rule used to derive rebalance dates."""

    name: str
    description: str
    offset: Optional[object] = None
    generator_func: Optional[Callable[[
        pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]] = None


class RebalPolicies:
    """Predefined U.S. business-month rebalancing calendars."""

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
    """Specification of a factor in the lens, including parents."""

    name: str
    ticker: str
    description: Optional[str] = None
    parents: List[str] = field(default_factory=list)
