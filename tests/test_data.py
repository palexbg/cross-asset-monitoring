from backend import utils

if __name__ == "__main__":
    ticker_symbol = ['SPY', 'QQQ', 'EEM', 'TLT',
                     'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
    a = utils.fetch_etf_data(ticker_symbol=ticker_symbol, end_date=None)
