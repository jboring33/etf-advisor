"""
logic/macro_overlay.py
======================
Macro Regime Overlay engine for applying top-down market conditions
to individual ETF tactical buy/sell signals.
"""

import pandas as pd
import numpy as np


def evaluate_market_regime(benchmark_series: pd.Series) -> dict:
    """
    Evaluates the top-down broad market regime using a benchmark index (e.g., SPY).
    Returns a dictionary with regime status, benchmark metrics, and signal notes.
    """
    s = benchmark_series.dropna()

    if len(s) < 200:
        return {
            "regime": "Neutral",
            "is_bullish": True,
            "sma200": 0.0,
            "latest_close": float(s.iloc[-1]) if not s.empty else 0.0,
            "note": "Insufficient benchmark history to evaluate macro regime."
        }

    latest_close = float(s.iloc[-1])
    sma200 = float(s.rolling(window=200).mean().iloc[-1])
    sma50 = float(s.rolling(window=50).mean().iloc[-1])

    if latest_close >= sma200 and sma50 >= sma200:
        regime = "Bullish"
        is_bullish = True
        note = f"Macro Regime: Bullish (Benchmark above 200 SMA: ${sma200:.2f})"
    elif latest_close < sma200:
        regime = "Bearish"
        is_bullish = False
        note = f"Macro Risk Warning: Bearish Regime (Benchmark below 200 SMA: ${sma200:.2f})"
    else:
        regime = "Neutral / Caution"
        is_bullish = True
        note = f"Macro Regime: Caution (Benchmark near 200 SMA: ${sma200:.2f})"

    return {
        "regime": regime,
        "is_bullish": is_bullish,
        "sma200": sma200,
        "latest_close": latest_close,
        "note": note
    }


def apply_macro_regime_overlay(etf_signal: dict, benchmark_series: pd.Series) -> dict:
    """
    Adjusts an individual ETF's composite score and tactical rating based on 
    the overall market regime.
    
    Rules in Bearish Macro Regime:
    - Deducts 15 points from composite score.
    - Caps tactical rating at 'Hold' (downgrades 'Strong Buy' or 'Buy').
    """
    macro_info = evaluate_market_regime(benchmark_series)
    
    # Create a copy so we don't mutate the raw tier 2 result directly
    adjusted_signal = etf_signal.copy()
    adjusted_signals_list = list(adjusted_signal.get("Signals", []))

    # Always document macro status in signals list
    adjusted_signals_list.append(macro_info["note"])

    # Apply penalty and cap rating if macro regime is bearish
    if not macro_info["is_bullish"]:
        raw_score = adjusted_signal.get("Composite_Score", 50.0)
        penalty_score = max(0.0, raw_score - 15.0)
        
        current_rating = adjusted_signal.get("Rating", "Hold")
        if current_rating in ["Strong Buy", "Buy"]:
            adjusted_rating = "Hold"
            adjusted_signals_list.append(
                f"Macro Overlay Constraint: Rating capped at 'Hold' due to Bearish Macro Regime."
            )
        else:
            adjusted_rating = current_rating

        adjusted_signal["Composite_Score"] = penalty_score
        adjusted_signal["Rating"] = adjusted_rating

    adjusted_signal["Signals"] = adjusted_signals_list
    adjusted_signal["Macro_Regime"] = macro_info["regime"]

    return adjusted_signal
