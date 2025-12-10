import pandas as pd
import numpy as np
from backend.backtester import BacktestResult
from backend.perfstats import PortfolioStats


def test_summary_table_returns_numeric_cagr():
    # Create a synthetic NAV series with ~0.1% daily returns over ~252 business days
    dates = pd.bdate_range(start="2024-01-01", periods=252)
    daily_ret = 0.001  # 0.1% per business day
    nav = 100.0 * (1 + daily_ret) ** np.arange(1, len(dates) + 1)
    nav_series = pd.Series(nav, index=dates)

    # minimal BacktestResult container fields
    holdings = pd.DataFrame(0.0, index=dates, columns=["A"])
    weights = pd.DataFrame(0.0, index=dates, columns=["A"])
    costs = pd.Series(0.0, index=dates)
    cash = pd.Series(0.0, index=dates)

    bt = BacktestResult(
        nav=nav_series,
        holdings=holdings,
        weights=weights,
        costs=costs,
        cash=cash,
        portfolio_assets_prices=pd.DataFrame(0.0, index=dates, columns=["A"])
    )

    rf = pd.Series(0.0, index=dates)

    ps = PortfolioStats(backtest_result=bt, risk_free=rf)
    tbl = ps.summary_table()

    # Expected CAGR computed using the same calendar-day annualization
    # logic used in `summary_table()` (uses calendar days between first
    # and last NAV and scales to 252 trading days).
    n_days = (dates[-1] - dates[0]).days
    expected = (nav_series.iloc[-1] /
                nav_series.iloc[0]) ** (252.0 / n_days) - 1.0

    # Find the CAGR metric
    cagr_row = tbl.loc[tbl["Metric"] == "Annualized Return (CAGR)", "Value"]
    assert not cagr_row.empty, "CAGR row missing from summary_table"
    cagr_value = float(cagr_row.iloc[0])

    # Allow 1% relative tolerance
    assert abs(cagr_value - expected) / expected < 0.01
