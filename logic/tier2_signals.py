"""
logic/tier2_signals.py
=======================
Tier 2 Tactical Scoring Engine: Calculates technical indicators (SMA, RSI, MACD),
evaluates momentum signals, and outputs composite 0-100 scores with actionable 
Buy / Hold / Sell ratings.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from config.portfolio import TIER2_INDICATOR_CONFIG


def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD Line, Signal Line, and MACD Histogram."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def evaluate_single_etf_technical_score(ticker, history_df=None):
    """
    Evaluates technical indicators for a single ETF and outputs a composite score (0-100).
    """
    if history_df is None or history_df.empty:
        try:
            t = yf.Ticker(ticker)
            history_df = t.history(period="1y")
        except Exception:
            return None

    if len(history_df) < 200:
        return None  # Needs enough history for 200 SMA calculation

    close = history_df["Close"]
    current_price = close.iloc[-1]

    # --- 1. CALCULATE TECHNICAL INDICATORS ---
    sma_cfg = TIER2_INDICATOR_CONFIG["Moving_Averages"]
    rsi_cfg = TIER2_INDICATOR_CONFIG["RSI"]
    macd_cfg = TIER2_INDICATOR_CONFIG["Volatility_MACD"]

    sma20 = close.rolling(window=sma_cfg["sma_short_period"]).mean().iloc[-1]
    sma50 = close.rolling(window=sma_cfg["sma_medium_period"]).mean().iloc[-1]
    sma200 = close.rolling(window=sma_cfg["sma_long_period"]).mean().iloc[-1]

    rsi_series = calculate_rsi(close, period=rsi_cfg["period"])
    current_rsi = rsi_series.iloc[-1]

    macd_line, signal_line, macd_hist = calculate_macd(
        close, macd_cfg["macd_fast"], macd_cfg["macd_slow"], macd_cfg["macd_signal"]
    )
    current_macd_hist = macd_hist.iloc[-1]
    prev_macd_hist = macd_hist.iloc[-2]

    # --- 2. COMPOSITE SCORING ENGINE (0 to 100 Points) ---
    score = 0
    signals = []

    # Trend Checks (Max 45 Pts)
    if current_price > sma200:
        score += sma_cfg["weights"]["price_above_200sma"]
        signals.append("Above 200-day SMA (Bullish primary trend)")
    else:
        signals.append("Below 200-day SMA (Bearish primary trend)")

    if sma50 > sma200:
        score += sma_cfg["weights"]["sma50_above_200sma"]
        signals.append("50 SMA above 200 SMA (Golden Cross alignment)")

    # RSI Momentum Checks (Max 30 Pts)
    if current_rsi <= rsi_cfg["oversold_threshold"]:
        score += 30
        signals.append(f"RSI Oversold ({round(current_rsi, 1)}) - Strong Buy opportunity")
    elif current_rsi < rsi_cfg["neutral_low"]:
        score += 20
        signals.append(f"RSI Bullish Neutral ({round(current_rsi, 1)})")
    elif rsi_cfg["neutral_low"] <= current_rsi <= rsi_cfg["neutral_high"]:
        score += 15
        signals.append(f"RSI Neutral ({round(current_rsi, 1)})")
    elif current_rsi >= rsi_cfg["overbought_threshold"]:
        score -= 10
        signals.append(f"RSI Overbought ({round(current_rsi, 1)}) - Extended valuation")

    # MACD Histogram Crossover Checks (Max 25 Pts)
    if current_macd_hist > 0 and prev_macd_hist <= 0:
        score += macd_cfg["macd_bullish_cross_weight"]
        signals.append("Bullish MACD Histogram Crossover")
    elif current_macd_hist > 0:
        score += 15
        signals.append("Positive MACD momentum")

    score = max(0, min(100, score))  # Clamp between 0 and 100

    # --- 3. DETERMINE ACTION RATING ---
    thresholds = TIER2_INDICATOR_CONFIG["Score_Thresholds"]
    if score >= thresholds["Strong Buy"]:
        rating = "Strong Buy"
    elif score >= thresholds["Buy"]:
        rating = "Buy"
    elif score >= thresholds["Hold"]:
        rating = "Hold"
    else:
        rating = "Sell"

    return {
        "Ticker": ticker,
        "Price ($)": round(current_price, 2),
        "Score": score,
        "Rating": rating,
        "RSI (14)": round(current_rsi, 1),
        "200 SMA": round(sma200, 2),
        "50 SMA": round(sma50, 2),
        "Key Signals": " | ".join(signals)
    }


def run_tier2_scoring(ticker_list):
    """
    Evaluates Tier 2 technical scores for a batch of tickers (e.g., your Favorited list).
    Returns a Pandas DataFrame sorted by composite score descending.
    """
    results = []
    for ticker in ticker_list:
        res = evaluate_single_etf_technical_score(ticker)
        if res:
            results.append(res)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return df_results
