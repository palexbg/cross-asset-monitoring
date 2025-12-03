from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd


@dataclass(frozen=True)
class RebalanceSchedule:
    name: str
    description: str
    offset: Optional[object] = None
    generator_func: Optional[Callable[[
        pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]] = None
