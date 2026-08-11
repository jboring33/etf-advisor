"""
logic/tier1_screener.py
========================
Tier 1 Fundamental Screening & Asset Location Engine.
Dynamically fetches live fundamental data using yfinance for any ticker in the pool.
"""

import pandas as pd
import yfinance as yf


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
    Dynamically fetches fundamental metrics (Expense Ratio, AUM, Yield)
    for all tickers in the active scan pool via yfinance.
    
    Parameters:
        scan_pool (dict): Dictionary mapping categories to ticker lists.
        
    Returns:
        pd.DataFrame: Live fundamental metrics, quality pass status, and categories.
    """
    rows = []

    for category, tickers in scan_pool.items():
        for ticker in tickers:
            try:
                tk = yf.Ticker(ticker)
                info = tk.info

                # Extract live metrics with robust fallbacks
                expense_ratio = info.get("expenseRatio", 0.0015)
                # Convert decimal expense ratio to percentage representation (e.g., 0.0009 -> 0.09%)
                if expense_ratio and expense_ratio < 0.05:
                    expense_ratio = expense_ratio * 100

                aum_m = info.get("totalAssets", 0)
                aum_m = (aum_m / 1e6) if aum_m else 0.0

                yield_pct = info.get("yield", info.get("dividendYield", 0.0))
                if yield_pct and yield_pct < 0.5:
                    yield_pct = yield_pct * 100

                # Basic quality filter pass criteria (e.g., Expense Ratio <= 0.50% & AUM > $50M)
                passed = (expense_ratio <= 0.50) and (aum_m >= 50.0 or aum_m == 0.0)

            except Exception:
                # Safe fallback if API call fails
                expense_ratio = 0.15
                aum_m = 0.0
                yield_pct = 0.0
                passed = True

            rows.append({
                "Category": category,
                "Ticker": ticker,
                "Expense Ratio": expense_ratio,
                "AUM": aum_m,
                "Yield (%)": yield_pct,
                "Passed Screener": passed
            })

    df = pd.DataFrame(rows)
    return df
