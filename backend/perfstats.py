"""
This a wrapper module around the QuantStats package to compute portfolio performance
statistics, reports, and plots. Most functions operate on Excess Returns,
however, QuantStats natively only accepts a fixed risk-free rate float, and 
hence others needed to be adapted here. QuantStats can provide a comprehensive
static html report if needed, function is implemented in get_html_report(), but 
it is not used for the dashboard UI.
"""


try:
    import quantstats as qs
except Exception:
    qs = None
import pandas as pd
import numpy as np

from .backtester import BacktestResult
from .utils import get_returns
from .structs import ReturnMethod
from typing import Optional
from backend.data import DataConfig


class PortfolioStats:
    """Lightweight wrapper around QuantStats for portfolio-level metrics."""

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
        if qs is None:
            raise ImportError("quantstats is required for calculate_stats()")
        output = qs.reports.metrics(
            self.excess_returns,
            mode=mode,
            rf=0.0,
            display=False
        )
        return output

    def summary_table(self) -> pd.DataFrame:
        """Return a compact DataFrame of key statistics suitable for UI display.

        This mirrors the small helper previously implemented in the UI layer.
        The method returns raw numeric values where appropriate (floats for
        percentage-like metrics and ints for day-counts). The UI layer is
        responsible for formatting presentation (percent vs. decimal).
        """
        try:
            stats = self.calculate_stats(mode="basic")
        except Exception:
            # If quantstats is unavailable or fails, continue with an empty
            # stats frame and fall back to NAV-derived metrics.
            stats = pd.DataFrame()

        nav = self.nav.dropna()
        rets = self.returns.dropna()

        if not nav.empty and not rets.empty:
            n_days = (nav.index[-1] - nav.index[0]).days
            if n_days > 0:
                cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n_days) - 1.0
            else:
                cagr = float("nan")
            ann_vol = rets.std() * (252 ** 0.5)
        else:
            cagr = float("nan")
            ann_vol = float("nan")

        metric_map = [
            ("Start Period", "Start Period"),
            ("End Period", "End Period"),
            ("Sharpe", "Sharpe"),
            ("Max drawdown", "Max Drawdown"),
            ("Longest drawdown (days)", "Longest DD Days"),
        ]

        rows = []
        for label, idx in metric_map:
            if idx in stats.index:
                value = stats.loc[idx].iloc[0]
                rows.append({"Metric": label, "Value": value})

        # Calendar and rolling returns from NAV
        if not nav.empty:
            last_date = nav.index[-1]

            # YTD
            start_ytd_date = pd.Timestamp(year=last_date.year, month=1, day=1)
            nav_ytd = nav[nav.index >= start_ytd_date]
            if len(nav_ytd) >= 2:
                ytd_ret = nav_ytd.iloc[-1] / nav_ytd.iloc[0] - 1.0
                rows.append({"Metric": "YTD", "Value": float(ytd_ret)})

            def _window_ret(window_days: int):
                if len(nav) < 2:
                    return None
                start_idx = max(0, len(nav) - window_days)
                sub = nav.iloc[start_idx:]
                if len(sub) >= 2:
                    return sub.iloc[-1] / sub.iloc[0] - 1.0
                return None

            three_m = _window_ret(63)
            if three_m is not None:
                rows.append({"Metric": "3M", "Value": float(three_m)})

            six_m = _window_ret(126)
            if six_m is not None:
                rows.append({"Metric": "6M", "Value": float(six_m)})

            one_y = _window_ret(252)
            if one_y is not None:
                rows.append({"Metric": "1Y", "Value": float(one_y)})

        # Append annualized metrics
        if pd.notna(cagr):
            rows.append({"Metric": "Annualized Return (CAGR)",
                        "Value": float(cagr)})
        if pd.notna(ann_vol):
            rows.append({"Metric": "Annualized Volatility",
                        "Value": float(ann_vol)})

        if rows:
            return pd.DataFrame(rows)

        # Fallback to whatever calculate_stats returned (if any)
        if not stats.empty:
            return stats.reset_index().rename(columns={"index": "Metric"})

        return pd.DataFrame()
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
