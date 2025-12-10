import numpy as np
import pandas as pd
import pytest
from backend.utils import normalize_prices_to_base_currency
from backend.structs import Currency, Asset


def test_fx_normalization_identity():
    """Test that normalizing USD prices to USD base returns identity."""
    prices = pd.DataFrame({
        'A': [100, 101, 102],
        'B': [200, 201, 202]
    }, index=pd.date_range('2020-01-01', periods=3))
    fx_data = pd.DataFrame({'USDUSD=X': [1, 1, 1]}, index=prices.index)
    # Asset metadata: both assets are USD-denominated
    assets = [Asset(name='A', asset_class='Eq', ticker='A', currency='USD'),
              Asset(name='B', asset_class='Eq', ticker='B', currency='USD')]
    out = normalize_prices_to_base_currency(
        prices, assets, fx_data, Currency.USD)
    pd.testing.assert_frame_equal(prices, out)


def test_fx_normalization_scaling():
    """Test that normalizing EUR prices to USD base divides by FX rate."""
    prices = pd.DataFrame({'A': [100, 200, 300]},
                          index=pd.date_range('2020-01-01', periods=3))
    fx_data = pd.DataFrame({'EURUSD=X': [2, 2, 2]}, index=prices.index)
    # Asset metadata: asset 'A' denominated in EUR and needs conversion to USD
    assets = [Asset(name='A', asset_class='Eq', ticker='A', currency='EUR')]
    out = normalize_prices_to_base_currency(
        prices, assets, fx_data, Currency.USD)
    # Should multiply by 2 (EUR -> USD using EURUSD rate = 2)
    expected = prices * 2
    pd.testing.assert_frame_equal(out.astype(float), expected.astype(float))
