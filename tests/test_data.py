from backend.perfstats import PortfolioStats
from backend.utils import dailify_riskfree, fetch_yfinance_data, get_returns, get_valid_rebal_vec_dates
from backend.moments import compute_ewma_covar
from pathlib import Path
from backend.factors import FactorConstruction, FactorExposure
from backend.structs import RebalPolicies, FactorAnalysisMode
from backend.config import BacktestConfig, FACTOR_LENS_UNIVERSE
from backend.backtester import run_backtest

import numpy as np
import pandas as pd

if __name__ == "__main__":
    investment_universe = ['SPY', 'QQQ', 'EEM', 'TLT',
                           'IEF', 'LQD', 'HYG', 'SHY', 'GLD', 'BND', '^IRX']

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
        data = fetch_yfinance_data(
            ticker_symbol=tickers_download,
            start_date='2015-12-31',
            end_date='2025-11-30')

    close = (data
             .loc[data["Field"] == 'Close']
             .set_index(['Date', 'Ticker'])
             ['Value']
             .unstack()
             )

    # Dailify risk free
    close, risk_free_rate = dailify_riskfree(close, ticker='^IRX')

    # for testing without RF
    risk_free_rate = pd.Series(data=0.0, index=close.index)

    factor_engine = FactorConstruction(prices=close[factor_universe],
                                       risk_free_rate=risk_free_rate,
                                       factor_definition=FACTOR_LENS_UNIVERSE)

    factors_ret = factor_engine.run()
    # or np.exp(factors_ret.cumsum(axis=0))
    factors_prices = (1 + factors_ret).cumprod(axis=0)

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
        rebal_vec=rebal_vec,
    )

    port_saa = PortfolioStats(
        backtest_result=saa, risk_free=risk_free_rate)
    perf_saa = port_saa.calculate_stats(mode='basic')

    port_saa.get_html_report(
        benchmark=None,
        title="SAA",
        output_filename="my_strategy_report.html"
    )

    print("Analyzing Exposures...")
    exposure_engine = FactorExposure(
        risk_factors=factors_prices,
        nav=port_saa.nav.to_frame(),
        risk_free_rate=risk_free_rate,
        analysis_mode=FactorAnalysisMode.ROLLING,
        lookback=120,
        smoothing_window=5
    )

    betas, t_stats, rsq = exposure_engine.run()

    return_attribution = exposure_engine.decompose_daily_returns(
        port_saa.nav, factors_prices)

    print('a')
