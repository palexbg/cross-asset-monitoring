import yfinance
import pandas as pd
import pdb


def fetch_etf_data(ticker_symbol: list[str],
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

    pdb.set_trace()
    return output.get(freq, f"Date conversion key not found, it must be one of {list(output.keys())}.")
