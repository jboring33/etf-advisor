"""
logic/tier2_signals.py
======================
Technical analysis engine for calculating tactical buy/sell signals on watchlisted ETFs.
Computes RSI(14), 50-day SMA, 200-day SMA, momentum scores, and composite tactical ratings.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def fetch_historical_prices(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Downloads historical adjusted closing prices for a list of ETF tickers.
    Returns a DataFrame where columns are Tickers and index is Date.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(tickers, period=period, progress=False)
        if "Adj Close" in data:
            prices = data["Adj Close"]
        elif "Close" in data:
            prices = data["Close"]
        else:
            prices = data

        # Ensure single ticker case returns a DataFrame with ticker column name
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        return prices
    except Exception as e:
        # Return empty DataFrame on fetch failure
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, window: int = 14) -> float:
    """
    Calculates the Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def calculate_tier2_signals(price_series: pd.Series, config: dict = None) -> dict:
    """
    Calculates moving averages, momentum metrics, and generates a composite 
    tactical rating (Strong Buy, Buy, Hold, Sell) along with individual signal notes.
    """
    # Clean series
    s = price_series.dropna()
    
    if len(s) < 50:
        return {
            "Close": float(s.iloc[-1]) if not s.empty else 0.0,
            "RSI": 50.0,
            "SMA50": 0.0,
            "SMA200": 0.0,
            "Composite_Score": 50.0,
            "Rating": "Hold",
            "Signals": ["Insufficient historical price data for complete technical analysis."]
        }

    latest_close = float(s.iloc[-1])
    
    # Calculate Technical Indicators
    rsi = calculate_rsi(s, window=14)
    sma50 = float(s.rolling(window=50).mean().iloc[-1]) if len(s) >= 50 else latest_close
    sma200 = float(s.rolling(window=200).mean().iloc[-1]) if len(s) >= 200 else sma50

    signals = []
    score = 50.0  # Neutral base score

    # Trend Checks
    if latest_close > sma50:
        score += 15
        signals.append("Price above 50-day SMA (Short-term Uptrend)")
    else:
        score -= 15
        signals.append("Price below 50-day SMA (Short-term Weakness)")

    if latest_close > sma200:
        score += 20
        signals.append("Price above 200-day SMA (Long-term Bullish Structure)")
    else:
        score -= 20
        signals.append("Price below 200-day SMA (Long-term Bearish Structure)")

    if sma50 > sma200:
        score += 15
        signals.append("Golden Cross Alignment (50-day SMA > 200-day SMA)")

    # RSI Checks
    if rsi < 30:
        score += 20
        signals.append(f"RSI strictly Oversold ({rsi:.1f}) — Rebound Opportunity")
    elif rsi > 70:
        score -= 15
        signals.append(f"RSI Overbought ({rsi:.1f}) — Consolidation Risk")
    else:
        signals.append(f"RSI Neutral ({rsi:.1f})")

    # Bound Composite Score between 0 and 100
    composite_score = float(np.clip(score, 0, 100))

    # Determine Rating Category
    if composite_score >= 75:
        rating = "Strong Buy"
    elif composite_score >= 60:
        rating = "Buy"
    elif composite_score >= 40:
        rating = "Hold"
    else:
        rating = "Sell"

    return {
        "Close": latest_close,
        "RSI": rsi,
        "SMA50": sma50,
        "SMA200": sma200,
        "Composite_Score": composite_score,
        "Rating": rating,
        "Signals": signals
    }
