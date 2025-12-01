from backend import utils
from backend.data import finTS, AssetSeries, IndicatorSeries
from backend.models import compute_ewma_covar
from pathlib import Path
import pandas as pd
import pdb

if __name__ == "__main__":
    ticker_symbol = ['SPY', 'QQQ', 'EEM', 'TLT',
                     'IEF', 'LQD', 'HYG', 'SHY', 'GLD']

    if Path('etf_data.csv').exists():
        data = pd.read_csv('etf_data.csv', parse_dates=['Date'])
    else:
        data = utils.fetch_etf_data(ticker_symbol=ticker_symbol, end_date=None)

    close = (data
             .loc[data["Field"] == 'Close']
             .set_index(['Date', 'Ticker'])
             ['Value']
             .unstack()
             )

    assets = AssetSeries(ticker=close.columns,
                         raw_data=close, freq=None)

    rets = assets.get_returns(freq=None)

    sigma_hat = compute_ewma_covar(returns=rets, span=21, annualize=True)
