from backend.utils import fetch_etf_data, get_returns
from backend.moments import compute_ewma_covar
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    investment_universe = ['SPY', 'QQQ', 'EEM', 'TLT',
                           'IEF', 'LQD', 'HYG', 'SHY', 'GLD']

    factor_tickers = {
        'ACWI': 'Equity',
        'GOVT': 'Rates',
        'HYG': 'Credit',
        'DBC': 'Commodities',
        'EEM': 'Emerging',
        'TIP': 'Inflation',
        'MTUM': 'Momentum',
        'VLUE': 'Value',
        'QUAL': 'Quality',
        'USMV': 'LowVol'
    }
    factor_universe = list(factor_tickers.keys())
    tickers_download = investment_universe + factor_universe

    if Path('etf_data.csv').exists():
        data = pd.read_csv('etf_data.csv', parse_dates=['Date'])
    else:
        data = fetch_etf_data(
            ticker_symbol=tickers_download,
            start_date='2015-12-31',
            end_date=None)

    close = (data
             .loc[data["Field"] == 'Close']
             .set_index(['Date', 'Ticker'])
             ['Value']
             .unstack()
             )

    rets = get_returns(close, lookback=1, type='log')

    instruments_ret = rets[investment_universe]
    factors_raw = rets[factor_universe].rename(factor_tickers, axis=1)

    sigma_hat = compute_ewma_covar(returns=rets, span=21, annualize=True)

    print('a')
