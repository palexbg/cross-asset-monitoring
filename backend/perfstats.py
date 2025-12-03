# Add a class here that will be a PortfolioObject with a NAV and a bunch of performance stats
# We just use quantstats to compute the stats and some summary reports
import quantstats as qs
import pandas as pd
from .backtester import BacktestResult
from .utils import get_returns
from typing import Optional


class PortfolioStats:
    def __init__(self, backtest_result: BacktestResult, risk_free: float = 0.0):
        self.backtest_result = backtest_result
        self.rf = risk_free / 100.0  # convert to decimal
        self.nav = backtest_result.nav
        self.returns = get_returns(self.nav)

    def calculate_stats(self, mode: str = 'basic') -> pd.DataFrame:
        output = qs.reports.metrics(
            self.returns,
            mode=mode,
            rf=self.rf,
            display=False
        )

        return output

    def get_html_report(self,
                        benchmark: pd.Series = None,
                        title='Strategy Tearsheet',
                        output_filename: Optional[str] = 'strat_report.html'
                        ) -> str:
        """
        Generates the full HTML report as a string.
        """

        output_mode = output_filename if output_filename else True

        result = qs.reports.html(
            self.returns,
            benchmark=benchmark,
            rf=self.rf,
            title=title,
            output=output_mode,
            download_filename=output_filename
        )

        if output_filename:
            print(f"Report saved to: {output_filename}")
            return None

        return result
