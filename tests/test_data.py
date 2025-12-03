from backend.perfstats import PortfolioStats
from backend.utils import fetch_etf_data, get_returns, get_valid_rebal_vec_dates
from backend.moments import compute_ewma_covar
from pathlib import Path
from backend.factors import FactorEngine
from backend.config import FACTOR_LENS_UNIVERSE, BacktestConfig, RebalPolicies
from backend.backtester import run_backtest
import quantstats as qs

import pandas as pd

if __name__ == "__main__":
    investment_universe = ['SPY', 'QQQ', 'EEM', 'TLT',
                           'IEF', 'LQD', 'HYG', 'SHY', 'GLD', 'BND']

    # should pull that from FACTOR_LENS_UNIVERSE
    factor_tickers = {
        'VT': 'Equity',
        'IEF': 'Rates',
        'LQD': 'Credit',
        'GSG': 'Commodities',
        # 'HYG': 'Credit2',
        # 'DBC': 'Commodities2',
        # 'EEM': 'Emerging',
        # 'TIP': 'Inflation',
        'VTV': 'Value',
        'MTUM': 'Momentum',
        # 'VLUE': 'Value2',
        # 'QUAL': 'Quality',
        # 'USMV': 'LowVol'
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

    rf = FactorEngine(prices=close[factor_universe],
                      config=FACTOR_LENS_UNIVERSE)

    factors_ret = rf.run()

    # test backtester
    portfolio_tickers = ['SPY', 'BND']
    pf_prices = close[portfolio_tickers]

    pf_weights = pd.DataFrame(
        index=pf_prices.index,
        columns=portfolio_tickers,
        data=0.0
    )

    valid_dates, rebal_vec = get_valid_rebal_vec_dates(
        schedule=RebalPolicies.US_MONTH_START,
        price_index=pf_prices.index
    )

    pf_weights.loc[valid_dates, 'SPY'] = 0.60
    pf_weights.loc[valid_dates, 'BND'] = 0.40

    saa = run_backtest(
        prices=pf_prices,
        target_weights=pf_weights,
        backtest_config=BacktestConfig(),
        rebal_vec=rebal_vec
    )

    port_saa = PortfolioStats(
        backtest_result=saa)
    perf_saa = port_saa.calculate_stats(mode='basic')

    port_saa.get_html_report(
        benchmark=None,
        title="SAA",
        output_filename="my_strategy_report.html"
    )

    # rets = get_returns(close, lookback=1, type='log')
    # instruments_ret = rets[investment_universe]

    # test one backtest on the SP500, just to code in the metrics and the portfolio analysis object

    # sigma_hat = compute_ewma_covar(returns=rets, span=21, annualize=True)

    # run_backtest(close)
    # weights = n
    # Backtester()

    print('a')
