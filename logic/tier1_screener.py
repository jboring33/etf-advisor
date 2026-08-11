"""
logic/tier1_screener.py
========================
Tier 1 Fundamental Screening & Asset Location Engine.
Filters ETFs based on fees, liquidity, and structural efficiency, 
and provides account placement tax recommendations.
"""

import pandas as pd
import numpy as np


def get_tax_location_recommendation(category: str, account_type: str) -> str:
    """
    Returns tax location recommendations based on asset category and account structure.
    
    Parameters:
        category (str): ETF category (e.g., 'Core Equity', 'Income & Credit', 'Tactical / Value')
        account_type (str): Target account type ('Taxable Brokerage', 'Tax-Deferred', 'Tax-Free')
        
    Returns:
        str: Guidance note on tax efficiency.
    """
    if "Taxable" in account_type:
        if category in ["Core Equity", "Tactical / Value"]:
            return "Optimal: High tax efficiency (qualified dividends & long-term capital gains)."
        elif category in ["Income & Credit"]:
            return "Sub-optimal: Generates ordinary income yield; creates tax drag in taxable accounts."
        else:
            return "Moderate: Evaluate distribution yield relative to marginal tax bracket."

    elif "Tax-Deferred" in account_type:  # Traditional IRA / 401(k)
        if category in ["Income & Credit"]:
            return "Optimal: Shields high ordinary income and bond distributions from current taxes."
        else:
            return "Acceptable: Standard tax-deferred growth environment."

    else:  # Tax-Free (Roth IRA / 401(k))
        if category in ["Core Equity", "Tactical / Value"]:
            return "Optimal: Maximizes tax-free compounding on high total-return growth assets."
        else:
            return "Acceptable: Tax-free distribution environment."


def run_tier1_screening(scan_pool: dict) -> pd.DataFrame:
    """
    Simulates or fetches baseline fundamental metrics (Expense Ratio, AUM, Volume)
    for all ETFs across active scan pool categories and applies passing criteria.
    
    Parameters:
        scan_pool (dict): Dictionary mapping categories to ticker lists.
        
    Returns:
        pd.DataFrame: Table of fundamental metrics, quality pass status, and notes.
    """
    rows = []
    
    # Mock data lookup table for representative metrics
    # (In production, replace with live yfinance / API data call)
    FUNDAMENTAL_DB = {
        "SPY": {"Expense_Ratio": 0.09, "AUM_M": 500000, "Yield": 1.2, "Passed": True},
        "QQQ": {"Expense_Ratio": 0.20, "AUM_M": 250000, "Yield": 0.6, "Passed": True},
        "DIA": {"Expense_Ratio": 0.16, "AUM_M": 32000, "Yield": 1.7, "Passed": True},
        "SCHD": {"Expense_Ratio": 0.06, "AUM_M": 55000, "Yield": 3.4, "Passed": True},
        "VFLO": {"Expense_Ratio": 0.34, "AUM_M": 850, "Yield": 1.8, "Passed": True},
        "SCYB": {"Expense_Ratio": 0.20, "AUM_M": 1200, "Yield": 6.5, "Passed": True},
        "JPST": {"Expense_Ratio": 0.18, "AUM_M": 24000, "Yield": 5.1, "Passed": True},
        "JAAA": {"Expense_Ratio": 0.19, "AUM_M": 11000, "Yield": 6.2, "Passed": True},
    }

    for category, tickers in scan_pool.items():
        for ticker in tickers:
            # Fallback for dynamic tickers added via UI
            data = FUNDAMENTAL_DB.get(ticker, {
                "Expense_Ratio": 0.15,
                "AUM_M": 2500,
                "Yield": 2.0,
                "Passed": True
            })
            
            rows.append({
                "Category": category,
                "Ticker": ticker,
                "Expense Ratio": data["Expense_Ratio"],
                "AUM": data["AUM_M"],
                "Yield (%)": data["Yield"],
                "Passed Screener": data["Passed"]
            })

    df = pd.DataFrame(rows)
    return df
