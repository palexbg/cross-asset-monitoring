from dataclasses import dataclass


@dataclass(frozen=True)
class config_backtest:
    initial_cash: float = 1_000_000.0
    cost_bps: float = 0.0010
    trade_at_close: bool = True
    reinvest: bool = True
    use_last_known_price: bool = True
    interest_rate: float = 0.0
    reinvest_proceeds: bool = True
