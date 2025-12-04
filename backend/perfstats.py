import quantstats as qs
import pandas as pd
import numpy as np
from .backtester import BacktestResult
from .utils import get_returns
from typing import Optional, Union


class PortfolioStats:
    def __init__(self,
                 backtest_result: BacktestResult,
                 risk_free: Union[float, pd.Series] = 0.0):

        self.backtest_result = backtest_result
        self.nav = backtest_result.nav

        # 1. Get Daily Returns
        self.returns = get_returns(self.nav, method='simple')

        # 2. Standardize Risk-Free Rate to a Series
        if isinstance(risk_free, pd.Series):
            # Align time-varying RF to the portfolio returns index
            self.rf_series = risk_free.reindex(self.returns.index).fillna(0.0)
        else:
            # Convert constant float (annual %) to daily decimal series
            rf_daily = (risk_free / 100.0) / 252.0
            self.rf_series = pd.Series(rf_daily, index=self.returns.index)

        # 3. Calculate Excess Returns (Pre-subtract RF)
        # This circumvents the QuantStats "Series is ambiguous" bug
        self.excess_returns = self.returns - self.rf_series

    def calculate_stats(self, mode: str = 'basic') -> pd.DataFrame:
        """
        Calculates metrics on Excess Returns.
        We pass rf=0.0 because we already subtracted self.rf_series.
        """
        output = qs.reports.metrics(
            self.excess_returns,
            mode=mode,
            rf=0.0,
            display=False
        )
        return output

    def get_html_report(self,
                        benchmark: pd.Series = None,
                        title='Strategy Tearsheet',
                        output_filename: Optional[str] = 'strat_report.html'
                        ) -> str:
        """
        Generates the full HTML report using Excess Returns.
        """
        # If benchmark is provided, we should also convert it to Excess Returns
        # for a fair "Apples to Apples" comparison (Alpha), though strictly
        # QuantStats usually takes raw benchmarks.
        # For now, we pass the raw benchmark, but note that Sharpe comparisons
        # will be (Strategy Excess) vs (Benchmark Raw).

        # Ideally, subtract RF from benchmark too if you want "Active vs Active":
        if benchmark is not None:
            # Align and subtract RF
            bench_ret = get_returns(benchmark).reindex(
                self.returns.index).fillna(0.0)
            bench_excess = bench_ret - self.rf_series
        else:
            bench_excess = None

        output_mode = output_filename if output_filename else True

        result = qs.reports.html(
            self.excess_returns,
            benchmark=bench_excess,
            rf=0.0,  # Crucial: Set to 0
            title=title + " (Excess Returns)",  # Label explicitly for clarity
            output=output_mode,
            download_filename=output_filename
        )

        if output_filename:
            print(f"Report saved to: {output_filename}")
            return None

        return result
