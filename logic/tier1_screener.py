"""
logic/tier1_screener.py
========================
Tier 1 Engine: Fetches live fundamental data, evaluates candidates against 
active risk profile limits, and assigns optimal account tax locations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from config.portfolio import DEFAULT_RISK_RULES, ACCOUNT_LOCATION_RULES

def fetch_etf_fundamentals(ticker_list):
    """
    Fetches real-time fundamental, fee, and risk metrics for a list of tickers.
    Uses yfinance ticker metadata with fallback calculations.
    """
    data = []
    
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Extract basic metadata
            name = info.get("shortName") or info.get("longName") or ticker
            category = info.get("category", "General Equity")
            
            # Fundamentals
            expense_ratio = (info.get("expenseRatio", 0.0) or 0.0) * 100
            div_yield = (info.get("yield") or info.get("dividendYield") or 0.0) * 100
            beta = info.get("beta", 1.0) or 1.0
            aum_m = (info.get("totalAssets") or 0) / 1e6
            price = info.get("previousClose") or t.fast_info.get("lastPrice", 0.0)
            
            # Calculate 3-Year Annualized Volatility from historical closes
            hist = t.history(period="3y")
            if not hist.empty and len(hist) > 100:
                daily_returns = hist['Close'].pct_change().dropna()
                volatility_3yr = (daily_returns.std() * np.sqrt(252)) * 100
            else:
                volatility_3yr = 15.0  # Default estimate if history unavailable
                
            data.append({
                "Ticker": ticker,
                "Name": name,
                "Category": category,
                "Price ($)": round(price, 2),
                "Expense (%)": round(expense_ratio, 2),
                "Yield (%)": round(div_yield, 2),
                "Beta": round(beta, 2),
                "AUM ($M)": round(aum_m, 1),
                "Vol 3Yr (%)": round(volatility_3yr, 2)
            })
        except Exception as e:
            # Skip invalid tickers or network errors gracefully
            continue
            
    return pd.DataFrame(data)


def assign_account_bucket(row):
    """
    Tax Location Strategy Engine:
    Assigns an ETF to an Account Bucket based on category, dividend yield, 
    and Foreign Tax Credit eligibility.
    """
    cat = str(row.get("Category", "")).lower()
    div_yield = row.get("Yield (%)", 0.0)
    
    # 1. TAXABLE BROKERAGE BUCKET
    # - Municipal Bonds (100% Federal Tax Exempt)
    # - Low-yield broad market core equities
    # - Developed & Emerging Markets ex-China (VEA, EMXC) -> Unlocks Foreign Tax Credit (Form 1116)
    if "muni" in cat or "developed markets" in cat or "emerging ex china" in cat or "foreign" in cat:
        if div_yield < ACCOUNT_LOCATION_RULES["Taxable Brokerage"]["max_income_yield"]:
            return "Taxable Brokerage"
            
    if ("large blend" in cat or "broad market" in cat or "total market" in cat) and div_yield < 2.5:
        return "Taxable Brokerage"

    # 2. TRADITIONAL / ROLLOVER IRA BUCKET (Income Shield)
    # - High ordinary income payouts (REITs, Corporate Bonds, High Yield, Options Income, CLOs)
    if ("bond" in cat or "reit" in cat or "real estate" in cat or 
        "high yield" in cat or "clo" in cat or "buywrite" in cat or 
        div_yield >= ACCOUNT_LOCATION_RULES["Traditional / Rollover IRA"]["min_income_yield"]):
        return "Traditional / Rollover IRA"

    # 3. ROTH IRA BUCKET (Tax-Free Capital Appreciation)
    # - High-growth, technology, semiconductors, high beta, dividend growth strategies
    return "Roth IRA"


def run_tier1_screen(df_fundamentals, risk_profile_name="Moderate"):
    """
    Evaluates fundamental data against active risk profile limits
    and tags passing ETFs with their target account bucket.
    """
    if df_fundamentals.empty:
        return pd.DataFrame()
        
    rules = DEFAULT_RISK_RULES.get(risk_profile_name, DEFAULT_RISK_RULES["Moderate"])
    
    # Apply Tier 1 Rule Filter Logic
    mask = (
        (df_fundamentals["Expense (%)"] <= rules["max_expense"]) &
        (df_fundamentals["Yield (%)"] >= rules["min_yield"]) &
        (df_fundamentals["Beta"] <= rules["max_beta"]) &
        (df_fundamentals["Vol 3Yr (%)"] <= rules["max_volatility_3yr"]) &
        (df_fundamentals["AUM ($M)"] >= rules["min_aum_m"])
    )
    
    df_passed = df_fundamentals[mask].copy()
    
    if not df_passed.empty:
        # Assign Account Location
        df_passed["Target Account"] = df_passed.apply(assign_account_bucket, axis=1)
        
    return df_passed
