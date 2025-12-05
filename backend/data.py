from abc import ABC, abstractmethod
import yfinance
import pandas as pd


class DataFetcher(ABC):
    @abstractmethod
    def fetch_close_prices(self, ticker_symbol: list[str],
                           start_date: str,
                           end_date: str,
                           store_data: bool = True) -> pd.DataFrame:
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
                 ).ffill(limit=5)

        if store_data:
            close.to_csv('etf_close_prices.csv', index=True)

        return close
