import numpy as np
import pandas as pd
import pytest
from backend.risk import AssetRiskEngine, FactorRiskEngine
from backend.config import AssetRiskConfig, FactorRiskConfig

# Dummy weights and prices for asset risk engine
prices = pd.DataFrame(np.random.rand(100, 3) + 100, columns=['A', 'B', 'C'])
weights = pd.DataFrame(1/3, index=prices.index, columns=prices.columns)


def test_asset_risk_contributions_sum_to_100():
    """Test that asset risk contributions sum to ~100%."""
    engine = AssetRiskEngine(weights, prices)
    rc = engine.run()
    ctr_pct = rc['latest_rc']['ctr_pct']
    total = ctr_pct.sum()
    assert np.isclose(total, 1.0, atol=0.05)  # Allow small numerical error


# Dummy betas and factor prices for factor risk engine
factor_prices = pd.DataFrame(np.random.rand(
    100, 2) + 100, columns=['Equity', 'Rates'])
betas = pd.DataFrame([[0.5, 0.5]], columns=['Equity', 'Rates'], index=[
                     factor_prices.index[-1]])


def test_factor_risk_contributions_sum_to_100():
    """Test that factor risk contributions sum to ~100%."""
    engine = FactorRiskEngine(betas, factor_prices)
    rc = engine.run()
    ctr_pct = rc['latest_factor_rc']['ctr_pct']
    total = ctr_pct.sum()
    assert np.isclose(total, 1.0, atol=0.05)
