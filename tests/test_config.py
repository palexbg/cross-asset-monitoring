import pytest
from backend.config import DataConfig, BacktestConfig, FactorConfig


def test_data_config_defaults():
    """Test DataConfig default values."""
    cfg = DataConfig()
    assert cfg.etf_data_path == 'cached_etf_close_prices.csv'
    assert cfg.maxfill_days == 5


def test_backtest_config_defaults():
    """Test BacktestConfig default values."""
    cfg = BacktestConfig()
    assert cfg.initial_cash == 1_000_000.0
    assert cfg.cost_rate == 0.0010
    assert cfg.trade_at_close is True


def test_factor_config_defaults():
    """Test FactorConfig default values."""
    cfg = FactorConfig()
    assert cfg.target_yearly_vol == 0.15
    assert cfg.scale_factors is True
