import pandas as pd
import pytest
from backend.structs import RebalanceSchedule, RebalPolicies
from backend.utils import get_valid_rebal_vec_dates


def test_us_month_start_rebal_dates():
    """Test that US month start schedule produces expected number of rebal dates."""
    idx = pd.date_range('2022-01-01', '2022-12-31', freq='B')
    schedule = RebalPolicies.US_MONTH_START
    dates, vec = get_valid_rebal_vec_dates(schedule, idx)
    # Should be 12 months in 2022
    assert len(dates) == 12
    assert vec.sum() == 12


def test_rebal_vec_alignment():
    """Test that rebal vector aligns with price index."""
    idx = pd.date_range('2022-01-01', '2022-03-31', freq='B')
    schedule = RebalPolicies.US_MONTH_END
    dates, vec = get_valid_rebal_vec_dates(schedule, idx)
    # All True values in vec should correspond to idx
    assert all(vec.index == idx)
    assert vec.sum() == 3  # Jan, Feb, Mar
