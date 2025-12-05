from backend.perfstats import PortfolioStats
from backend.utils import dailify_riskfree, fetch_yfinance_data, get_returns, get_valid_rebal_vec_dates
from backend.moments import compute_ewma_covar
from pathlib import Path
from backend.factors import FactorConstruction, FactorExposure
from backend.structs import RebalPolicies, FactorAnalysisMode, Asset
from backend.config import BacktestConfig, FACTOR_LENS_UNIVERSE
from backend.backtester import run_backtest
from backend.risk import AssetRiskEngine

import numpy as np
import pandas as pd

if __name__ == "__main__":

    investment_universe = [Asset(name='SPY', Asset_Class='Equity', ticker='SPY', description='S&P 500 ETF'),
                           Asset(name='IEF', Asset_Class='Rates', ticker='IEF',
                                 description='7-10 Year Treasury ETF'),
                           Asset(name='LQD', Asset_Class='Credit', ticker='LQD',
                                 description='Investment Grade Corporate Bond ETF'),
                           Asset(name='HYG', Asset_Class='Credit', ticker='HYG',
                                 description='High Yield Corporate Bond ETF'),
                           Asset(name='SHY', Asset_Class='Rates', ticker='SHY',
                                 description='1-3 Year Treasury ETF'),
                           Asset(name='GLD', Asset_Class='Commodities',
                                 ticker='GLD', description='Gold ETF'),
                           Asset(name='BND', Asset_Class='Bond', ticker='BND',
                                 description='Total Bond Market ETF'),
                           Asset(name='^IRX', Asset_Class='Rates', ticker='^IRX', description='13 Week Treasury Bill')]

    # should pull that from FACTOR_LENS_UNIVERSE
    factor_tickers = {
        'VT': 'EquityFactor',
        'IEF': 'RatesFactor',
        'LQD': 'CreditFactor',
        'GSG': 'CommoditiesFactor',
        # 'HYG': 'Credit2',
        # 'DBC': 'Commodities2',
        # 'EEM': 'Emerging',
        # 'TIP': 'Inflation',
        'VTV': 'ValueFactor',
        'MTUM': 'MomentumFactor',
        # 'VLUE': 'Value2',
        # 'QUAL': 'Quality',
        # 'USMV': 'LowVol'
    }

    factor_universe = list(factor_tickers.keys())
    tickers_download = [
        asset.ticker for asset in investment_universe] + factor_universe

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
    # risk_free_rate = pd.Series(data=0.0, index=close.index)

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

    # Tab 1:
    port_saa = PortfolioStats(
        backtest_result=saa, risk_free=risk_free_rate)
    perf_saa = port_saa.calculate_stats(mode='basic')

    port_saa.get_html_report(
        benchmark=None,
        title="SAA",
        output_filename="my_strategy_report.html"
    )

    # Tab 2:
    print("Analyzing Factor Exposures...")
    exposure_engine = FactorExposure(
        risk_factors=factors_prices,
        nav=port_saa.nav.to_frame(),
        risk_free_rate=risk_free_rate,
        analysis_mode=FactorAnalysisMode.ROLLING,
        lookback=120,
        smoothing_window=5
    )

    betas, t_stats, rsq, resid = exposure_engine.run()

    print("Analyzing Return contributions...")
    return_attribution = exposure_engine.decompose_daily_returns(
        port_saa.nav, factors_prices)

    print("Analyzing Risk contributions...")
    risk_contribution = AssetRiskEngine(
        weights=saa.weights,
        prices=saa.portfolio_assets_prices,
        rebal_vec=rebal_vec,
        compute_over_time=True,
        annualize=True
    ).run()

    print("Analyzing Factor Risk contributions...")

    print('a')
