from backend.perfstats import PortfolioStats
from backend.utils import get_valid_rebal_vec_dates, build_index_from_returns

from pathlib import Path
from backend.factors import FactorConstruction, FactorExposure
from backend.structs import RebalPolicies, FactorAnalysisMode, Asset, Currency, ReturnMethod
from backend.config import BacktestConfig, FACTOR_LENS_UNIVERSE
from backend.backtester import run_backtest
from backend.risk import AssetRiskEngine
from backend.data import YFinanceDataFetcher, UniverseLoader
from backend.risk import FactorRiskEngine
import pandas as pd
import numpy as np

if __name__ == "__main__":

    portfolio_base_ccy = Currency.USD  # or Currency.EUR / Currency.CHF

    investment_universe_metadata = [Asset(name='SPY', asset_class='Equity', ticker='SPY', description='S&P 500 ETF', currency='USD'),
                                    Asset(name='IEF', asset_class='Rates', ticker='IEF', currency='USD',
                                          description='7-10 Year Treasury ETF'),
                                    Asset(name='LQD', asset_class='Credit', ticker='LQD', currency='USD',
                                          description='Investment Grade Corporate Bond ETF'),
                                    Asset(name='HYG', asset_class='Credit', ticker='HYG', currency='USD',
                                          description='High Yield Corporate Bond ETF'),
                                    Asset(name='SHY', asset_class='Rates', ticker='SHY', currency='USD',
                                          description='1-3 Year Treasury ETF'),
                                    Asset(name='GLD', asset_class='Commodities', currency='USD',
                                          ticker='GLD', description='Gold ETF'),
                                    Asset(name='BND', asset_class='Bond', ticker='BND', currency='USD',
                                          description='Total Bond Market ETF'),
                                    Asset(name='^IRX', asset_class='Rates', ticker='^IRX',
                                          description='13 Week Treasury Bill', currency='USD'),
                                    Asset(name='EL4W.DE', asset_class='Rates', ticker='EL4W.DE', currency='EUR',
                                          description='Germany Money Market'),
                                    Asset(name='CSBGC3.SW', asset_class='Rates', ticker='CSBGC3.SW', currency='CHF',
                                          description='Swiss Domestic Short Term Bond')]

    fx_universe_metadata = [Asset(name='EURUSD=X', asset_class='FX', ticker='EURUSD=X',
                                  description='Euro to US Dollar Exchange Rate', currency='USD'),
                            Asset(name='CHFUSD=X', asset_class='FX', ticker='CHFUSD=X',
                                  description='Swiss Franc to US Dollar Exchange Rate', currency='USD'),
                            Asset(name='CHFEUR=X', asset_class='FX', ticker='CHFEUR=X',
                                  description='Swiss Franc to Euro Exchange Rate', currency='EUR')]

    factor_universe_metadata = [Asset(name='VT', asset_class='EquityFactor', ticker='VT', description='Vanguard Total World Stock ETF', currency='USD'),
                                Asset(name='IEF', asset_class='RatesFactor', ticker='IEF',
                                      description='iShares 7-10 Year Treasury Bond ETF', currency='USD'),
                                Asset(name='LQD', asset_class='CreditFactor', ticker='LQD',
                                      description='iShares iBoxx $ Investment Grade Corporate Bond ETF', currency='USD'),
                                Asset(name='GSG', asset_class='CommoditiesFactor', ticker='GSG',
                                      description='iShares S&P GSCI Commodity-Indexed Trust', currency='USD'),
                                Asset(name='VTV', asset_class='ValueFactor', ticker='VTV',
                                      description='Vanguard Value Index Fund ETF Shares', currency='USD'),
                                Asset(name='MTUM', asset_class='MomentumFactor', ticker='MTUM',
                                      description='iShares MSCI USA Momentum Factor ETF', currency='USD'),
                                Asset(name='UDN', asset_class='FXFactor', ticker='UDN',
                                      description='Adjusted Invesco DB US Dollar Index Bearish Fund', currency='USD')]

    asset_tickers = list(
        {asset.ticker: asset for asset in investment_universe_metadata}.keys())
    factor_tickers = list(
        {asset.ticker: asset for asset in factor_universe_metadata}.keys())
    fx_tickers = list(
        {asset.ticker: asset for asset in fx_universe_metadata}.keys())

    tickers_download = list(set(asset_tickers + factor_tickers + fx_tickers))

    data_API = YFinanceDataFetcher()
    universe_loader = UniverseLoader(data_API)
    close, risk_free_rate = universe_loader.load_or_fetch_universe(
        close_csv_path=Path('etf_close_prices.csv'),
        investment_universe=investment_universe_metadata,
        factor_universe=factor_universe_metadata,
        fx_universe=fx_universe_metadata,
        base_currency=portfolio_base_ccy,
        start_date='2015-12-31',
        end_date='2025-11-30',
    )

    # for testing without RF
    # risk_free_rate = pd.Series(data=0.0, index=close.index)

    factor_engine = FactorConstruction(prices=close[factor_tickers],
                                       risk_free_rate=risk_free_rate,
                                       factor_definition=FACTOR_LENS_UNIVERSE)

    factors_ret = factor_engine.run()
    factors_prices = build_index_from_returns(
        factors_ret, method=ReturnMethod.LOG)

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
    asset_risk = AssetRiskEngine(
        weights=saa.weights,
        prices=saa.portfolio_assets_prices,
        rebal_vec=rebal_vec,
        annualize=True
    ).run()

    print(asset_risk['latest_rc'])
    print(asset_risk['latest_port_vol'])

    print("Analyzing Factor Risk contributions...")

    factor_risk = FactorRiskEngine(
        betas=betas,                     # from FactorExposure.run()
        factor_prices=factors_prices,    # you already build this
        residual_var=resid,              # add later once you store it
        rebal_vec=rebal_vec,             # optional
        annualize=True
    ).run()

    print(factor_risk["latest_factor_rc"])
    print(factor_risk["latest_systematic_vol"])
    print(factor_risk["latest_idio_vol"])
