"""
test_signals.py
===============
Unit tests for Tier 2 technical indicator calculations and Macro Overlay logic.
Placed directly in the project root directory.
"""

import sys
import os

# Guarantee project root is in sys.path for direct root execution
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import pandas as pd
import numpy as np

from logic.tier2_signals import calculate_rsi, calculate_tier2_signals
from logic.macro_overlay import evaluate_market_regime, apply_macro_regime_overlay


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def bullish_price_series():
    """Generates a 250-day steadily upward trending price series."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
    prices = np.linspace(100.0, 200.0, 250)
    return pd.Series(prices, index=dates)


@pytest.fixture
def bearish_price_series():
    """Generates a 250-day steadily downward trending price series."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
    prices = np.linspace(200.0, 100.0, 250)
    return pd.Series(prices, index=dates)


@pytest.fixture
def short_price_series():
    """Generates an insufficient 30-day price series."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
    prices = np.linspace(100.0, 110.0, 30)
    return pd.Series(prices, index=dates)


# ==============================================================================
# TIER 2 SIGNALS TESTS
# ==============================================================================

def test_calculate_rsi_edge_cases():
    """Verifies RSI calculation under normal and flat/zero-loss conditions."""
    # Flat line (no price gain or loss) -> diff is zero
    dates = pd.date_range(end=pd.Timestamp.today(), periods=20, freq="D")
    flat_series = pd.Series([100.0] * 20, index=dates)
    rsi_flat = calculate_rsi(flat_series, window=14)
    assert rsi_flat == 50.0  # Fallback neutral when loss/gain is zero

    # Pure upward gain (zero loss)
    gaining_series = pd.Series(np.linspace(100, 150, 20), index=dates)
    rsi_gain = calculate_rsi(gaining_series, window=14)
    assert rsi_gain > 70.0


def test_tier2_signals_short_history(short_price_series):
    """Verifies fallback result when price history is less than 50 days."""
    res = calculate_tier2_signals(short_price_series)
    assert res["Rating"] == "Hold"
    assert res["Composite_Score"] == 50.0
    assert "Insufficient historical price data" in res["Signals"][0]


def test_tier2_signals_bullish_trend(bullish_price_series):
    """Verifies that a clear uptrend generates a Strong Buy or Buy rating."""
    res = calculate_tier2_signals(bullish_price_series)
    assert res["Rating"] in ["Strong Buy", "Buy"]
    assert res["Composite_Score"] >= 60.0
    assert res["Close"] > res["SMA50"]
    assert res["SMA50"] > res["SMA200"]


# ==============================================================================
# MACRO OVERLAY TESTS
# ==============================================================================

def test_evaluate_market_regime(bullish_price_series, bearish_price_series):
    """Verifies broad market regime detection for bullish vs. bearish trends."""
    bull_regime = evaluate_market_regime(bullish_price_series)
    assert bull_regime["is_bullish"] is True
    assert bull_regime["regime"] == "Bullish"

    bear_regime = evaluate_market_regime(bearish_price_series)
    assert bear_regime["is_bullish"] is False
    assert bear_regime["regime"] == "Bearish"


def test_apply_macro_regime_overlay_bearish_cap(bullish_price_series, bearish_price_series):
    """
    Verifies that a candidate ETF with a Strong Buy signal is capped at 'Hold'
    and penalized by 15 points when the macro benchmark (e.g. SPY) is bearish.
    """
    # Generate raw ETF signal (Strong Buy)
    raw_signal = calculate_tier2_signals(bullish_price_series)
    initial_score = raw_signal["Composite_Score"]
    
    # Apply macro overlay with a BEARISH benchmark
    adjusted_signal = apply_macro_regime_overlay(raw_signal, bearish_price_series)

    # Composite score should decrease by 15 points
    assert adjusted_signal["Composite_Score"] == max(0.0, initial_score - 15.0)
    
    # Rating must be capped at Hold
    assert adjusted_signal["Rating"] == "Hold"
    
    # Macro risk signal note should be present
    assert any("Macro Overlay Constraint" in s for s in adjusted_signal["Signals"])
