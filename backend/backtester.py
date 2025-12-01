import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from numba import njit
from dataclasses import dataclass
from .config import BacktestConfig

# TODO
# check if the size of rebal_vec, prices and weights are all proper
# also the dates needs to match
# and the weights need to sum up to 1 at every point
# check if rebal_vec contains booleans
# add trading at open prices (combinations - close to open, open to open, close to close, open to close)
# add long/short/gross/net legs too.


@dataclass
class BacktestResult():
    nav: pd.Series
    holdings: pd.DataFrame
    weights: pd.DataFrame
    costs: pd.Series
    cash: pd.Series

# Perhaps one can optimize this further by just using numpy, but it is nice to have the dates
# Also additional objects from here would be interesting to extract, I woudl say.
# So a dataclass would make A LOT more sense (a backtesting object)
# Assume that we only trade at close now

# This should be located elsewhere


def validate_inputs(
    prices: pd.DataFrame,
    weights: pd.DataFrame
) -> None:

    if not prices.index.equals(weights.index):
        raise ValueError("Index Mismatch: Align data before backtesting.")

    if prices.values.isna().any().any():
        raise ValueError("NaNs in Prices.")

    if weights.values.isna().any().any():
        raise ValueError("NaNs in Weights.")

    if prices.values.isinf().any().any():
        raise ValueError("Infinity in Prices.")

    if weights.values.isinf().any().any():
        raise ValueError("Infinity in Weights.")

    if prices.shape != weights.shape:
        raise ValueError(f"Shape mismatch: {prices.shape} vs {weights.shape}")

    if (weights.values < 0).any().any():
        print("Warning: Negative weights.")


def run_backtest(
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        rebal_freq: Optional[str] = None,
        rebal_vec: Union[pd.DataFrame, pd.Series] = None,
        backtest_config: BacktestConfig = BacktestConfig()
) -> BacktestResult:

    validate_inputs(prices=prices, weights=target_weights)

    if rebal_vec is not None:
        is_rebal_day = rebal_vec.to_numpy()
    elif rebal_freq is not None:
        pass
    else:  # try to infer it from the target weights. Rebalancing is whenever all bigger than 0
        weight_delta = target_weights.diff().abs().sum(axis=1)
        is_rebal_day = (weight_delta > 1e-6).to_numpy(dtype=bool)

    # Numba works with numpy arrays, convert everything beforehand
    nav, holdings, weights, cash, costs = backtest_kernel(
        prices=prices.to_numpy(dtype=np.float64),
        target_weights=target_weights.to_numpy(dtype=np.float64),
        is_rebal_day=is_rebal_day,
        initial_cash=backtest_config.initial_cash,
        trade_at_close=backtest_config.trade_at_close,
        reinvest_proceeds=backtest_config.reinvest_proceeds,
        transaction_costs_bps=backtest_config.cost_bps,
        use_last_known_price=backtest_config.use_last_known_price,
        interest_cash=backtest_config.interest_rate
    )

    output = BacktestResult(
        nav=pd.Series(nav, index=prices.index, name='NAV'),
        holdings=pd.DataFrame(holdings, index=prices.index,
                              columns=prices.columns),
        weights=pd.DataFrame(weights, index=prices.index,
                             columns=weights.columns),
        costs=pd.Series(costs, index=prices.index),
        cash=pd.Series(cash, index=prices.index)
    )

    return output


@njit
def backtest_kernel(
    prices: np.ndarray,
    target_weights: np.ndarray,
    # decouple signal from execution
    is_rebal_day: np.ndarray = None,
    initial_cash: float = 1_000_000.0,

    # Execution logic
    transaction_costs_bps: float = 0.0,

    # -- this needs to be fixed and accounted for
    trade_at_close: bool = True,
    reinvest_proceeds: bool = True,
    use_last_known_price: bool = True,
    interest_cash: float = 0.0

) -> Tuple[np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:

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
        current_prices = prices[t]

        # Mark-to-market, daily - this is before trading
        portfolio_value = np.sum(current_prices * current_holdings)
        current_nav = portfolio_value + current_cash

        # we are trading at close, best we could do is to allocate the shares compared to yesterday's prices and NAV from yesterday.
        if is_rebal_day[t]:
            if t == 0:
                sizing_nav = initial_cash
                # if we invest on the first day we have a hack and assume that we trade after the actual close
                sizing_prices = prices[0]
            else:
                # we aim to guard against implementation shortfall for any rebal day, other than the first one
                # see https://www.investopedia.com/terms/i/implementation-shortfall.asp
                sizing_nav = nav_history[t-1]
                sizing_prices = prices[t-1]

            target_val = sizing_nav * target_weights[t]

            target_holdings = target_val / sizing_prices

            # we compare the new trade units at t, with the last ones known at t-1
            trade_units = target_holdings - current_holdings

            # we trade at the actual price in t, see difference to sizing prices. This is to reduce implementation shortfall
            trade_value = np.sum(trade_units * current_prices)
            # this is always positive, so we take the transaction costs out of the cash
            transaction_cost = np.sum(
                np.abs(trade_units) * current_prices) * transaction_costs_bps

            # Store the history
            # - update cash
            cash_history[t] = current_cash - transaction_cost - trade_value

            # - update the costs
            costs_history[t] = transaction_cost

            # This is what actually happened
            current_holdings = target_holdings

            # Update recurrencies
            current_cash = cash_history[t]
            current_holdings = holdings_history[t]
        else:
            costs_history[t] = 0.0

            # cash we do not touch for now

        # store history
        nav_history[t] = current_nav
        cash_history[t] = current_cash
        holdings_history[t] = current_holdings

        weights_history[t, :] = (current_holdings * current_prices)/current_nav

    return nav_history, holdings_history, weights_history, cash_history, costs_history
