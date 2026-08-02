"""
logic/tier1_screener.py
=======================
Engine for fetching ETF fundamental data via yfinance, executing Tier 1 threshold 
screening based on dynamic risk profiles, and mapping funds to tax-efficient 
account locations.
"""

import yfinance as yf
import pandas as pd
import numpy as np

def fetch_etf_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Fetches fundamental metrics (Expense Ratio, Yield, Beta, AUM, Volatility) 
    for a given list of ETF tickers using yfinance.
    """
    if not tickers:
        return pd.DataFrame()

    data_list = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Extract fundamental fields with safe fallback defaults
            expense_ratio = info.get("expenseRatio", info.get("annualReportExpenseRatio", 0.0015))
            if expense_ratio is not None and expense_ratio < 0.05:  # Handle decimal vs percentage representation
                expense_ratio = expense_ratio * 100

            div_yield = info.get("dividendYield", info.get("yield", 0.0))
            if div_yield is not None and div_yield < 0.15:
                div_yield = div_yield * 100

            beta = info.get("beta", info.get("beta3Year", 1.0))
            if beta is None:
                beta = 1.0

            aum = info.get("totalAssets", info.get("marketCap", 500000000))
            aum_m = (aum / 1e6) if aum else 500.0

            # Calculate historical volatility if 3-year history is available
            hist = t.history(period="3y")
            if not hist.empty and len(hist) > 20:
                volatility_3yr = float(hist["Close"].pct_change().std() * np.sqrt(252) * 100)
            else:
                volatility_3yr = 15.0

            data_list.append({
                "Ticker": ticker,
                "Name": info.get("shortName", info.get("longName", ticker)),
                "Category": info.get("category", "General ETF"),
                "Expense_Ratio": float(expense_ratio) if expense_ratio is not None else 0.15,
                "Dividend_Yield": float(div_yield) if div_yield is not None else 1.5,
                "Beta": float(beta),
                "AUM_M": float(aum_m),
                "Volatility_3Yr": float(volatility_3yr)
            })
        except Exception:
            # Fallback mock row in case yfinance rate-limits or fails for a symbol
            data_list.append({
                "Ticker": ticker,
                "Name": f"{ticker} (Data Unavailable)",
                "Category": "Unknown",
                "Expense_Ratio": 0.20,
                "Dividend_Yield": 1.50,
                "Beta": 1.00,
                "AUM_M": 250.0,
                "Volatility_3Yr": 16.0
            })

    return pd.DataFrame(data_list)


def run_tier1_screen(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Screens fundamental ETF DataFrame against dynamic numerical thresholds passed from 
    st.session_state (Max Expense, Min Yield, Max Beta, Min AUM, Max Volatility).
    """
    if df.empty:
        return df

    filtered_df = df[
        (df["Expense_Ratio"] <= rules.get("max_expense", 1.50)) &
        (df["Dividend_Yield"] >= rules.get("min_yield", 0.00)) &
        (df["Beta"] <= rules.get("max_beta", 2.50)) &
        (df["AUM_M"] >= rules.get("min_aum_m", 0)) &
        (df["Volatility_3Yr"] <= rules.get("max_volatility_3yr", 60.0))
    ].copy()

    return filtered_df


def map_account_location(row: pd.Series) -> str:
    """
    Determines optimal account location (Taxable Brokerage, Roth IRA, Traditional IRA)
    based on dividend yield, expected income generation, and volatility characteristics.
    """
    div_yield = row.get("Dividend_Yield", 0.0)
    category = str(row.get("Category", "")).lower()

    # Fixed income, high dividend, REITs, and covered call strategies -> Deferred/Exempt
    if div_yield > 3.5 or "bond" in category or "reit" in category or "real estate" in category:
        return "Traditional IRA"
    
    # High capital appreciation / growth -> Roth IRA
    elif div_yield <= 1.5 and row.get("Beta", 1.0) >= 1.0:
        return "Roth IRA"
    
    # Low yield, broad index funds, and highly tax-efficient equity funds -> Taxable
    else:
        return "Taxable Brokerage"
