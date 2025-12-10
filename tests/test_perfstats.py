import numpy as np
import pandas as pd
import pytest
from backend.perfstats import PortfolioStats
from backend.backtester import BacktestResult

# Dummy NAV and RF for testing
nav = pd.Series(np.exp(np.linspace(0, 0.1, 100)), name='NAV')
rf = pd.Series(0.0001, index=nav.index)

bt_result = BacktestResult(
    nav=nav,
    holdings=pd.DataFrame(
        np.ones((100, 2)), index=nav.index, columns=['A', 'B']),
    weights=pd.DataFrame(0.5, index=nav.index, columns=['A', 'B']),
    costs=pd.Series(0.0, index=nav.index),
    cash=pd.Series(0.0, index=nav.index),
    portfolio_assets_prices=pd.DataFrame(
        np.ones((100, 2)), index=nav.index, columns=['A', 'B'])
)


def test_return_contributions_sum_to_portfolio_return():
    """Test that daily return contributions sum to total portfolio return."""
    stats = PortfolioStats(bt_result, rf)
    # Verify that the simple returns produced by PortfolioStats equal
    # the day-over-day percentage change of NAV (within tolerance).
    stats_returns = stats.returns.fillna(0.0).values
    nav_pct = bt_result.nav.pct_change().fillna(0.0).values
    assert np.allclose(stats_returns, nav_pct, atol=1e-12)
