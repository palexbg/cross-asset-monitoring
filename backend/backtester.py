"""Backtest engine for long-only portfolios with target weights.

Provides a simple long-only backtester built around a Numba
``backtest_kernel`` plus a ``BacktestResult`` container for NAV,
holdings, weights, cash and costs time series.
"""

import pandas as pd
import numpy as np
import warnings

from typing import Optional, Tuple
from numba import njit
from dataclasses import dataclass
from .config import BacktestConfig


@dataclass
class BacktestResult():
    """Container for backtest outputs such as NAV, holdings and cash."""

    nav: pd.Series
    holdings: pd.DataFrame
    weights: pd.DataFrame
    costs: pd.Series
    cash: pd.Series
    portfolio_assets_prices: pd.DataFrame

    def gross_exposure(self) -> pd.Series:
        """Return portfolio gross exposure over time (not yet implemented)."""
        raise NotImplementedError("Gross exposure is not implemented yet")

    def turnover(self) -> pd.Series:
        """Return portfolio turnover over time (not yet implemented)."""
        raise NotImplementedError("Turnover not implemented yet")


def validate_inputs(
    prices: pd.DataFrame,
    weights: pd.DataFrame
) -> None:
    """Validate that input price and weight matrices are consistent and clean."""

    if not prices.index.equals(weights.index):
        raise ValueError("Index Mismatch: Align data before backtesting.")

    if prices.isna().values.any():
        raise ValueError("NaNs in Prices.")

    if weights.isna().values.any():
        raise ValueError("NaNs in Weights.")

    if np.isinf(prices.values).any():
        raise ValueError("Infinity in Prices.")

    if np.isinf(weights.values).any():
        raise ValueError("Infinity in Weights.")

    if prices.shape != weights.shape:
        raise ValueError(f"Shape mismatch: {prices.shape} vs {weights.shape}")

    if (weights < 0).any().any():
        warnings.warn("Warning: Negative weights.", RuntimeWarning)


def run_backtest(
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        backtest_config: BacktestConfig,
        rebal_freq: Optional[str] = None,
        rebal_vec: Optional[pd.Series] = None
) -> BacktestResult:
    """Run a simple long-only backtest driven by target weights.

    High-level flow
    ---------------
    1. Validate that ``prices`` and ``target_weights`` share the same
       index, shape and are free of NaNs/inf.
    2. Convert the boolean ``rebal_vec`` series into a NumPy mask
       marking trade dates (typically rebalance dates).
    3. Pass prices, target weights and the rebalancing mask into the
       Numba ``backtest_kernel``, together with configuration such as
       initial cash, transaction costs and execution flags.
    4. Wrap the resulting NAV, holdings, weights, costs and cash series
       into a ``BacktestResult`` dataclass for further analysis.
    """

    validate_inputs(prices=prices, weights=target_weights)

    if rebal_vec is not None:
        if not np.issubdtype(rebal_vec.dtype, np.bool_):
            raise TypeError('rebal_vec must be boolean')
        is_rebal_day = rebal_vec.to_numpy()
    else:
        raise NotImplementedError(
            "Calendar based rebal is not implemented yet")

    # Numba works with numpy arrays, convert everything beforehand
    nav, holdings, weights, cash, costs = backtest_kernel(
        prices=prices.to_numpy(dtype=np.float64),
        target_weights=target_weights.to_numpy(dtype=np.float64),
        is_rebal_day=is_rebal_day,
        initial_cash=backtest_config.initial_cash,
        trade_at_close=backtest_config.trade_at_close,
        reinvest_proceeds=backtest_config.reinvest_proceeds,
        transaction_costs_bps=backtest_config.cost_rate,
        use_last_known_price=backtest_config.use_last_known_price,
        interest_cash=backtest_config.interest_rate
    )

    output = BacktestResult(
        nav=pd.Series(nav, index=prices.index, name='NAV'),
        holdings=pd.DataFrame(holdings, index=prices.index,
                              columns=prices.columns),
        weights=pd.DataFrame(weights, index=prices.index,
                             columns=prices.columns),
        costs=pd.Series(costs, index=prices.index),
        cash=pd.Series(cash, index=prices.index),
        portfolio_assets_prices=prices
    )

    return output


@njit
def backtest_kernel(
    prices: np.ndarray,
    target_weights: np.ndarray,
    # decouple signal from execution
    is_rebal_day: np.ndarray,
    initial_cash: float = 1_000_000.0,

    # Execution logic
    transaction_costs_bps: float = 0.0,

    # -- not implemented yet --
    trade_at_close: bool = True,
    reinvest_proceeds: bool = True,
    use_last_known_price: bool = True,
    interest_cash: float = 0.0

) -> Tuple[np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:
    """Numba-accelerated core backtest loop for a long-only portfolio.

     High-level behaviour
     --------------------
     Simulates the evolution of a cash + long-only asset portfolio given:

     - a matrix of asset prices ``prices[t, i]``,
     - a matrix of target portfolio weights ``target_weights[t, i]``, and
     - a boolean rebalancing mask ``is_rebal_day[t]``.

     At each time step ``t`` the kernel:

     1. Marks the current portfolio to market using ``prices[t, :]`` and
         the current holdings to obtain a pre-trade NAV.
     2. If ``is_rebal_day[t]`` is True, computes desired target holdings
         in *units* from the target weights and a "sizing" NAV/prices
         pair (yesterday's NAV and prices, except on the first day where
         it uses ``initial_cash`` and today's prices).
     3. Converts the gap between current and target holdings into trade
         units, values those trades at today's prices, and computes
         transaction costs as a bps charge on traded notional.
     4. Updates cash by subtracting both the trade value and transaction
         costs, and updates holdings by applying the trade units.
     5. Re-marks the portfolio to market after trades to obtain the
         end-of-day NAV, and records NAV, cash, holdings, costs and the
         implied portfolio weights time series.

     The execution flags (``trade_at_close``, ``reinvest_proceeds``,
     ``use_last_known_price``, ``interest_cash``) are accepted for future
     extension but not yet used; the current logic assumes trades execute
     at the close, proceeds are fully reinvested, and cash does not earn
     interest.
     """

    T, N = prices.shape

    holdings_history = np.zeros((T, N))  # this is in units (fractional shares)
    weights_history = np.zeros((T, N))  # this is in percentage
    nav_history = np.zeros(T)
    cash_history = np.zeros(T)
    costs_history = np.zeros(T)

    # initialization
    # we will be carrying the cash and putting the daily pnl to it
    # likewise, we will be monitoring the holdings s the weights
    current_cash = initial_cash
    current_holdings = np.zeros(N)  # we do not hold anything at the beginning

    for t in range(T):
        prices_t = prices[t, :]

        # Mark-to-market, daily - this is before trading
        portfolio_value = np.sum(prices_t * current_holdings)
        current_nav = portfolio_value + current_cash

        transaction_cost = 0.0

        # we are trading at close, best we could do is to allocate the shares compared to yesterday's prices and NAV from yesterday.
        if is_rebal_day[t]:
            if t == 0:
                sizing_nav = initial_cash
                # if we invest on the first day we have a hack and assume that we trade after the actual close
                sizing_prices = prices_t
            else:
                # we aim to guard against implementation shortfall for any rebal day, other than the first one
                # see https://www.investopedia.com/terms/i/implementation-shortfall.asp
                # actual deviation of targeted holdings vs executed holdings are not stored yet
                sizing_nav = nav_history[t - 1]
                sizing_prices = prices[t - 1, :]

            target_val = sizing_nav * target_weights[t, :]

            target_holdings = target_val / sizing_prices

            # we compare the new trade units at t, with the last ones known at t-1
            trade_units = target_holdings - current_holdings

            # we trade at the actual price in t, see difference to sizing prices. This is to reduce implementation shortfall
            trade_value = np.sum(trade_units * prices_t)

            # this is always positive, so we take the transaction costs out of the cash
            transaction_cost = np.sum(
                np.abs(trade_units) * prices_t) * transaction_costs_bps

            # - update cash
            current_cash = current_cash - transaction_cost - trade_value

            current_holdings = current_holdings + trade_units

        current_nav = np.sum(prices_t * current_holdings) + current_cash

        nav_history[t] = current_nav
        cash_history[t] = current_cash
        holdings_history[t] = current_holdings
        costs_history[t] = transaction_cost

        # update the weights
        if current_nav != 0.0:
            weights_history[t, :] = (current_holdings * prices_t)/current_nav
        else:
            weights_history[t, :] = 0.0

        # some warnings
        # if current_cash < 0.0:
        #     print('Warning: Cash is negative')
        if current_nav < 0.0:
            raise ValueError("NAV is negative")
        # if np.any(current_holdings < 0.0):
        #     print("Warning: Holdings contain negative positions")

    return nav_history, holdings_history, weights_history, cash_history, costs_history
