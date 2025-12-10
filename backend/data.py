from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple

import yfinance
import pandas as pd

from backend.structs import Asset, Currency
from backend.utils import normalize_prices_to_base_currency, dailify_risk_free
from backend.factors import triangulate_fx_factor
from backend.config import DataConfig


class DataFetcher(ABC):
    """Abstract interface for fetching historical close prices from a data API."""

    @abstractmethod
    def fetch_close_prices(self, ticker_symbol: list[str],
                           start_date: str,
                           end_date: str,
                           store_data: bool = True) -> pd.DataFrame:
        """Return a DataFrame of close prices for the requested tickers."""
        pass


class YFinanceDataFetcher(DataFetcher):
    def fetch_close_prices(self, ticker_symbol: list[str],
                           start_date: str = '2015-12-31',
                           end_date: str = '2024-11-30',
                           store_data: bool = True) -> pd.DataFrame:
        """
        Fetches historical ETF data for the given ticker symbols.
        Args:
        ticker_symbol (list[str]): List of ETF ticker symbols.
        start_date (str): Start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): End date for fetching data in 'YYYY-MM-DD' format.
        store_data (bool): Whether to store the fetched data as a CSV file.
        Returns: pandas.DataFrame: DataFrame containing the historical ETF data.
        """

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

        close = (data
                 .loc[data["Field"] == 'Close']
                 .set_index(['Date', 'Ticker'])
                 ['Value']
                 .unstack()
                 )

        if store_data:
            close.to_csv('etf_close_prices.csv', index=True)

        return close


class UniverseLoader:
    """Helper that prepares an asset, factor and FX price universe.

    Uses a ``DataFetcher`` implementation to either load cached data
    or download prices on demand, then normalizes to base currency and
    attaches a daily risk-free rate.
    """

    def __init__(self, data_api: DataFetcher):
        """Create a loader bound to a particular data API implementation."""
        self.data_api = data_api

    def load_or_fetch_universe(
        self,
        close_csv_path: Path,
        investment_universe: Sequence[Asset],
        factor_universe: Sequence[Asset],
        fx_universe: Sequence[Asset],
        base_currency: Currency,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load or fetch the full price universe and risk-free series.

        Steps
        -----
        1. Derive the unique tickers for investment assets, factor ETFs
           and FX pairs from the provided metadata.
        2. If a CSV file exists, load it and optionally top it up with
           any missing tickers via the configured ``DataFetcher``;
           otherwise, download all required tickers from scratch.
        3. Forward-fill missing observations up to ``DataConfig.maxfill_days``
           to smooth over short gaps in the raw data.
        4. Convert asset prices into the chosen base currency using
           FX pairs and ``normalize_prices_to_base_currency``.
        5. Attach a daily simple risk-free return series via
           ``dailify_risk_free``.
        6. For the FX factor, construct a base-currency factor index via
           ``triangulate_fx_factor`` and append it to the price panel.

        Returns
        -------
        close : DataFrame
            All prices (assets + factors + FX factor) indexed by Date.
        risk_free_rate : Series
            Daily risk-free rate aligned to ``close.index``.
        """

        inv_meta = list(investment_universe)
        fac_meta = list(factor_universe)
        fx_meta = list(fx_universe)

        asset_tickers = list({a.ticker: a for a in inv_meta}.keys())
        factor_tickers = list({a.ticker: a for a in fac_meta}.keys())
        fx_tickers = list({a.ticker: a for a in fx_meta}.keys())

        tickers_download = list(
            set(asset_tickers + factor_tickers + fx_tickers))

        if close_csv_path.exists():
            close = pd.read_csv(close_csv_path, parse_dates=[
                                "Date"]).set_index("Date")
            missing = [t for t in tickers_download if t not in close.columns]
            if missing:
                close = self.data_api.fetch_close_prices(
                    ticker_symbol=tickers_download,
                    start_date=start_date,
                    end_date=end_date,
                )
        else:
            close = self.data_api.fetch_close_prices(
                ticker_symbol=tickers_download,
                start_date=start_date,
                end_date=end_date,
            )

        # Forward fill missing data
        close.ffill(limit=DataConfig.maxfill_days, inplace=True)

        # Normalize to base currency
        close = normalize_prices_to_base_currency(
            close_data=close,
            asset_metadata=inv_meta,
            fx_data=close[fx_tickers] if fx_tickers else None,
            base_currency=base_currency,
        )

        # Dailify risk free
        close, risk_free_rate = dailify_risk_free(
            close, base_currency=base_currency
        )

        # FX factor preconstruction
        fx_factor_assets = [a for a in fac_meta if a.asset_class == "FXFactor"]
        fx_factor_ticker = fx_factor_assets[0].name if fx_factor_assets else None
        if fx_factor_ticker:
            fx_factor_prices = triangulate_fx_factor(
                fx_data=close[fx_tickers],
                base_currency=base_currency,
                ccy_factor_data=close[fx_factor_ticker],
            )
            close[fx_factor_ticker] = fx_factor_prices

        return close, risk_free_rate
