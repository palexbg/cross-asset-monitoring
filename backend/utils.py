from typing import Tuple

import pandas as pd
import numpy as np
import warnings

from .structs import RebalanceSchedule, Asset, ReturnMethod, Currency


def get_returns(
    prices: pd.DataFrame,
    lookback: int = 1,
    method: ReturnMethod | str = ReturnMethod.LOG
) -> pd.DataFrame:
    # Normalize method to ReturnMethod enum for safety
    if isinstance(method, str):
        try:
            method = ReturnMethod(method)
        except ValueError:
            raise ValueError("method must be either 'log' or 'simple'")

    if method == ReturnMethod.LOG:
        returns = np.log(prices/prices.shift(lookback))
    elif method == ReturnMethod.SIMPLE:
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


def dailify_risk_free(
    prices: pd.DataFrame,
    base_currency: Currency | str,
    days_in_year: int = 252
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Hard-coded risk-free selection and processing by base currency.

    Assumptions:
      USD -> ^IRX     (annualized % yield index)
      EUR -> EL4W.DE  (money market ETF price series)
      CHF -> CSBGC3.SW (short Swiss gov ETF price series)

    Output:
      Adds 'RiskFreeRate' as a daily simple return series.
    """

    if isinstance(base_currency, str):
        try:
            ccy = Currency(base_currency.upper().strip())
        except ValueError:
            raise ValueError("base_currency must be one of: USD, EUR, CHF")
    else:
        ccy = base_currency
    out = prices.copy()

    if ccy == Currency.USD:
        # ^IRX is an annualized percent yield
        rf = (out["^IRX"] / 100.0) / days_in_year

    elif ccy == Currency.EUR:
        # ETF price -> daily simple return
        rf = out["EL4W.DE"].pct_change()

    elif ccy == Currency.CHF:
        # ETF price -> daily simple return
        rf = out["CSBGC3.SW"].pct_change()
    else:
        raise ValueError("base_currency must be one of: USD, EUR, CHF")

    rf = rf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rf.name = "RiskFreeRate"
    out["RiskFreeRate"] = rf

    return out, rf


def normalize_prices_to_base_currency(
    close_data: pd.DataFrame,  # includes everything
    # includes just the assets or whatever needs to potentially be converted. Everything is Asset python dataclass
    asset_metadata: list[Asset],
    fx_data: pd.DataFrame,
    base_currency: Currency | str = Currency.USD
) -> pd.DataFrame:
    """
    Converts asset prices to portfolio base currency.
    """
    normalized_prices = close_data.copy()

    # Normalize base_currency to string for comparison with asset.currency
    if isinstance(base_currency, Currency):
        base_ccy_str = base_currency.value
    else:
        base_ccy_str = str(base_currency)

    # Assets in foreign currency
    foreign_assets = [a for a in asset_metadata if a.currency != base_ccy_str]

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
