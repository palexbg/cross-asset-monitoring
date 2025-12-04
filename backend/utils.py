from typing import Tuple
import yfinance
import pandas as pd
import numpy as np
import warnings

from .structs import RebalanceSchedule


def fetch_yfinance_data(ticker_symbol: list[str],
                        start_date: str = '2015-12-31',
                        end_date: str = '2024-12-31',
                        store_data: bool = True) -> pd.DataFrame:
    """
    Fetches historical ETF data for the given ticker symbols.
    Args:
        ticker_symbol (list[str]): List of ETF ticker symbols.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        store_data (bool): Whether to store the fetched data as a CSV file.
        Returns: pandas.DataFrame: DataFrame containing the historical ETF data."""

    data = yfinance.download(tickers=ticker_symbol,
                             start=start_date,
                             end=end_date,
                             ignore_tz=True,
                             auto_adjust=True)

    data = (
        data
        .stack(level=1, future_stack=True)
        .reset_index()
        .melt(
            id_vars=['Date', 'Ticker'],
            var_name='Field',
            value_name='Value',
        )
    )

    if store_data:
        data.to_csv('etf_data.csv', index=False)

    return data


def freq2days(freq):

    output = {
        "B": 252,
        "M": 12,
        "D": 365.25,
        "W": 52
    }

    return output.get(freq, f"Date conversion key not found, it must be one of {list(output.keys())}.")


def get_returns(prices: pd.DataFrame, lookback: int = 1, method: str = 'log') -> pd.DataFrame:
    # TODO: freq has to be a pandas thingy, need to give the list of those here
    # Resample the data to the desired frequency, use pandas offset aliases

    if method == 'log':
        returns = np.log(prices/prices.shift(lookback))
    elif method == 'simple':
        returns = prices/prices.shift(lookback) - 1
    else:
        raise ValueError("method must be either 'log' or 'simple'")

    if np.isnan(returns.values).any():
        # base class for warnings about dubious runtime behavior
        warnings.warn(
            "There are remaining NaNs in the series", RuntimeWarning)
    return returns


def get_valid_rebal_vec_dates(schedule: RebalanceSchedule, price_index: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, pd.Series]:
    """
    Generate valid trading (rebal) dates. New schedules have to be added in the RebalanceSchedule class.
    """
    start_date = price_index[0]
    end_date = price_index[-1]

    # Generate theoretical rebal dates based on prices
    if schedule.generator_func:
        theoretical_dates = schedule.generator_func(start_date, end_date)
    elif schedule.offset:
        theoretical_dates = pd.date_range(
            start=start_date, end=end_date, freq=schedule.offset)
    else:
        raise ValueError(
            f"Schedule {schedule.name} is missing both offset and generator_func")

    # Prices intersects actual rebal dates
    # Validate that (!)
    valid_dates = price_index.searchsorted(theoretical_dates)
    valid_dates = price_index.intersection(theoretical_dates)

    # generate the rebal vec with true and false
    rebal_vec = pd.Series(False, index=price_index)
    rebal_vec.loc[valid_dates] = True

    return valid_dates, rebal_vec
