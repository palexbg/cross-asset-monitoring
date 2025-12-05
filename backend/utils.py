from typing import Tuple

import pandas as pd
import numpy as np
import warnings

from .structs import RebalanceSchedule, Asset


def normalize_prices_to_base_currency(
    prices: pd.DataFrame,
    asset_metadata: list[Asset],
    fx_data: pd.DataFrame,
    base_currency: str = 'USD'
) -> pd.DataFrame:
    """
    Converts asset prices to portfolio base currency using provided FX data.
    Handles direct (EURUSD) and inverse (1/USDEUR) rate conventions.
    """
    normalized_prices = prices.copy()

    # Filter for assets that need to be converted
    foreign_assets = [a for a in asset_metadata if a.currency != base_currency]

    if not foreign_assets:
        return normalized_prices

    for asset in foreign_assets:
        # 1. Try Direct Pair (e.g. Asset=EUR, Base=USD -> EURUSD=X)
        direct_pair = f"{asset.currency}{base_currency}=X"
        # 2. Try Inverse Pair (e.g. Asset=USD, Base=EUR -> EURUSD=X)
        inverse_pair = f"{base_currency}{asset.currency}=X"

        if direct_pair in fx_data.columns:
            # Price_Base = Price_Local * (Local/Base)
            fx_rate = fx_data[direct_pair]
            normalized_prices[asset.ticker] = prices[asset.ticker] * fx_rate

        elif inverse_pair in fx_data.columns:
            # Price_Base = Price_Local * (1 / (Base/Local))
            fx_rate = fx_data[inverse_pair]
            # Avoid division by zero in FX data
            fx_rate = fx_rate.replace(0.0, float('nan'))
            normalized_prices[asset.ticker] = prices[asset.ticker] / fx_rate

        else:
            # Senior Dev Move: Log a warning or raise specific error, don't fail silently
            raise ValueError(f"Missing FX pair for {asset.ticker}: "
                             f"Need {direct_pair} or {inverse_pair}")

    return normalized_prices


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
    valid_dates = price_index.intersection(theoretical_dates)

    # generate the rebal vec with true and false
    rebal_vec = pd.Series(False, index=price_index)
    rebal_vec.loc[valid_dates] = True

    return valid_dates, rebal_vec


def dailify_riskfree(prices: pd.DataFrame, ticker: str = '^IRX') -> pd.DataFrame:
    if ticker in prices.columns:
        prices[ticker] = (prices[ticker] / 100.0) / 252.0
        prices.rename(columns={ticker: 'RiskFreeRate'}, inplace=True)
    else:
        warnings.warn(
            f"Ticker {ticker} not found in prices columns.", RuntimeWarning)

    return prices, prices['RiskFreeRate']


def normalize_prices_to_base_currency(
    close_data: pd.DataFrame,  # includes everything
    # includes just the assets or whatever needs to potentially be converted. Everything is Asset python dataclass
    asset_metadata: list[Asset],
    fx_data: pd.DataFrame,
    base_currency: str = 'USD'
) -> pd.DataFrame:
    """
    Converts asset prices to portfolio base currency.
    """
    normalized_prices = close_data.copy()

    # Assets in foreign currency
    foreign_assets = [a for a in asset_metadata if a.currency != base_currency]

    if not foreign_assets:
        return normalized_prices

    for asset in foreign_assets:

        direct_pair = f"{asset.currency}{base_currency}=X"
        inverse_pair = f"{base_currency}{asset.currency}=X"

        if direct_pair in fx_data.columns:
            # Price_Base = Price_Local * (Local/Base)
            fx_rate = fx_data[direct_pair]
            normalized_prices[asset.ticker] = close_data[asset.ticker] * fx_rate

        elif inverse_pair in fx_data.columns:

            fx_rate = fx_data[inverse_pair]

            # if there is zero, better to produce a nan
            fx_rate = fx_rate.replace(0.0, float('nan'))

            normalized_prices[asset.ticker] = close_data[asset.ticker] / fx_rate
        else:
            # raise a flag if a pair is missing
            raise ValueError(f"Missing FX pair for {asset.ticker}: "
                             f"Need {direct_pair} or {inverse_pair}")

    return normalized_prices
