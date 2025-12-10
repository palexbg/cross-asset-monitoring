"""
This a wrapper module around the QuantStats package to compute portfolio performance
statistics, reports, and plots. Most functions operate on Excess Returns,
however, QuantStats natively only accepts a fixed risk-free rate float, and 
hence others needed to be adapted here. QuantStats can provide a comprehensive
static html report if needed, function is implemented in get_html_report(), but 
it is not used for the dashboard UI.
"""


import quantstats as qs
import pandas as pd
import numpy as np

from .backtester import BacktestResult
from .utils import get_returns
from .structs import ReturnMethod
from typing import Optional
from backend.data import DataConfig


class PortfolioStats:
    def __init__(self,
                 backtest_result: BacktestResult,
                 risk_free: pd.Series
                 ):

        self.backtest_result = backtest_result
        self.nav = backtest_result.nav

        self.returns = get_returns(self.nav, method=ReturnMethod.SIMPLE)
        self.rf_series = risk_free.reindex(self.returns.index).fillna(
            0.0, limit=DataConfig.maxfill_days)

        # excess returns as quantstats handles only a fixed float
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

    # --- Plot helpers returning matplotlib figures for embedding ---

    def plot_monthly_heatmap_fig(self):
        """Return a matplotlib Figure with the monthly returns heatmap.
        """

        r = self.excess_returns.copy()

        r = r.loc[self.nav.index.min(): self.nav.index.max()]
        r = r.dropna()

        fig = qs.plots.monthly_heatmap(r, show=False)

        fig = fig.get_figure()
        return fig

    def plot_drawdown_fig(self):
        """Return a matplotlib Figure with the drawdown curve."""

        fig = qs.plots.drawdown(self.excess_returns, show=False)

        fig = fig.get_figure()

        return fig

    def get_drawdown_series(self) -> pd.Series:
        """Return drawdown series for use in Streamlit charts.

        This mirrors QuantStats' drawdown calculation but exposes the
        underlying series so that we can plot it with Altair.
        """
        # QuantStats' stats.to_drawdown_series gives a drawdown time series
        dd = qs.stats.to_drawdown_series(self.excess_returns)
        return dd

    def plot_rolling_vol_fig(self, window: int = 126):
        """Return a matplotlib Figure with rolling volatility (window in days)."""

        fig = qs.plots.rolling_volatility(
            self.excess_returns, period=window, show=False)

        fig = fig.get_figure()
        return fig

    def get_rolling_vol_series(self, window: int = 126) -> pd.Series:
        """Return rolling volatility series for use in Streamlit charts.

        This mirrors the logic of the rolling vol figure but exposes the
        underlying annualized volatility time series so we can plot it
        with native Streamlit charts (matching heights with st.line_chart).
        """
        r = self.excess_returns.copy().dropna()

        # Daily returns rolling std scaled to annual vol with sqrt(252)
        rv = r.rolling(window=window).std() * np.sqrt(252)
        return rv

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

        if benchmark is not None:
            # Align and subtract RF
            bench_ret = get_returns(benchmark, method=ReturnMethod.SIMPLE).reindex(
                self.returns.index).fillna(0.0)
            bench_excess = bench_ret - self.rf_series
        else:
            bench_excess = None

        output_mode = output_filename if output_filename else True

        result = qs.reports.html(
            self.excess_returns,
            benchmark=bench_excess,
            rf=0.0,
            title=title + " (Excess Returns)",
            output=output_mode,
            download_filename=output_filename
        )

        if output_filename:
            print(f"Report saved to: {output_filename}")
            return None

        return result
