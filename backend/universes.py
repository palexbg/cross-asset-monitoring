from __future__ import annotations

"""Definitions of investable asset, factor and FX universes used in the app. For now it uses only
Yahoo Finance tickers."""

from typing import List

from .structs import Asset


def get_investment_universe() -> List[Asset]:
    """Return the list of investable assets for portfolio construction."""

    return [
        Asset(name="SPY", asset_class="Equity", ticker="SPY",
              description="S&P 500 ETF", currency="USD"),
        Asset(name="IEF", asset_class="Rates", ticker="IEF",
              description="7-10 Year Treasury ETF", currency="USD"),
        Asset(name="LQD", asset_class="Credit", ticker="LQD",
              description="Investment Grade Corporate Bond ETF", currency="USD"),
        Asset(name="HYG", asset_class="Credit", ticker="HYG",
              description="High Yield Corporate Bond ETF", currency="USD"),
        Asset(name="SHY", asset_class="Rates", ticker="SHY",
              description="1-3 Year Treasury ETF", currency="USD"),
        Asset(name="GLD", asset_class="Commodities", ticker="GLD",
              description="Gold ETF", currency="USD"),
        Asset(name="BND", asset_class="Bond", ticker="BND",
              description="Vanguard Total Bond Market ETF", currency="USD"),
        Asset(name="^IRX", asset_class="Rates", ticker="^IRX",
              description="13 Week Treasury Bill", currency="USD"),
        Asset(name="EL4W.DE", asset_class="Rates", ticker="EL4W.DE",
              description="Germany Money Market", currency="EUR"),
        Asset(name="CSBGC3.SW", asset_class="Rates", ticker="CSBGC3.SW",
              description="Swiss Domestic Short Term Bond", currency="CHF"),
    ]


def get_factor_universe() -> List[Asset]:
    """Universe of raw ETFs used to build factor indices."""

    return [
        Asset(name="VT", asset_class="EquityFactor", ticker="VT",
              description="Vanguard Total World Stock ETF", currency="USD"),
        Asset(name="IEF", asset_class="RatesFactor", ticker="IEF",
              description="iShares 7-10 Year Treasury Bond ETF", currency="USD"),
        Asset(name="LQD", asset_class="CreditFactor", ticker="LQD",
              description="iShares iBoxx $ Investment Grade Corporate Bond ETF", currency="USD"),
        Asset(name="GSG", asset_class="CommoditiesFactor", ticker="GSG",
              description="iShares S&P GSCI Commodity-Indexed Trust", currency="USD"),
        Asset(name="VTV", asset_class="ValueFactor", ticker="VTV",
              description="Vanguard Value Index Fund ETF Shares", currency="USD"),
        Asset(name="MTUM", asset_class="MomentumFactor", ticker="MTUM",
              description="iShares MSCI USA Momentum Factor ETF", currency="USD"),
        Asset(name="UDN", asset_class="FXFactor", ticker="UDN",
              description="Adjusted Invesco DB US Dollar Index Bearish Fund", currency="USD"),
    ]


def get_fx_universe() -> List[Asset]:
    """FX pairs required for triangulation of the FX factor and currency conversion."""

    return [
        Asset(name="EURUSD=X", asset_class="FX", ticker="EURUSD=X",
              description="EURUSD", currency="USD"),
        Asset(name="CHFUSD=X", asset_class="FX", ticker="CHFUSD=X",
              description="CHFUSD", currency="USD"),
        Asset(name="CHFEUR=X", asset_class="FX", ticker="CHFEUR=X",
              description="CHFEUR", currency="EUR"),
    ]
