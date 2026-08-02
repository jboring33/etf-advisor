"""
config/portfolio.py
===================
Centralized configuration file storing ETF ticker universes, risk profiles, 
and technical scoring parameters.
"""

# Universe Pool — Expanded baseline for comprehensive screening
DYNAMIC_SCAN_POOL = {
    "Core Equity": [
        "VOO", "VTI", "SCHD", "VYM", "DGRO", "IVV", 
        "USMV", "SPLV"
    ],
    "Growth & Tech": [
        "QQQ", "VUG", "IYW", "SMH", "SCHG"
    ],
    "Income & Credit": [
        "JAAA", "JEPI", "JEPQ", "SCYB", "JPST", "BND", 
        "SGOV", "BIL", "SHY", "VCIT", "MUB"
    ],
    "International ex-China": [
        "VEA", "EMXC", "VXUS", "ACWV"
    ]
}

ALL_SCAN_TICKERS = list(set([ticker for list_ in DYNAMIC_SCAN_POOL.values() for ticker in list_]))

DEFAULT_FAVORITES = ["VOO", "SCHD", "QQQ", "JAAA", "VEA"]

# Adjusted Baseline Risk Profiles
DEFAULT_RISK_RULES = {
    "Conservative": {
        "description": "Capital preservation, low volatility, high quality.",
        "max_expense": 0.25,
        "min_yield": 0.5,        # Adjusted lower to allow broad low-vol index screening
        "max_beta": 0.85,
        "min_aum_m": 500,
        "max_volatility_3yr": 16.0
    },
    "Moderate": {
        "description": "Balanced growth and income with moderate market exposure.",
        "max_expense": 0.35,
        "min_yield": 0.0,
        "max_beta": 1.15,
        "min_aum_m": 250,
        "max_volatility_3yr": 22.0
    },
    "Aggressive": {
        "description": "Maximum growth potential; higher beta and momentum tolerance.",
        "max_expense": 0.75,
        "min_yield": 0.0,
        "max_beta": 2.00,
        "min_aum_m": 50,
        "max_volatility_3yr": 40.0
    }
}

ACCOUNT_LOCATION_RULES = {
    "Taxable Brokerage": {"max_income_yield": 2.5},
    "Traditional / Rollover IRA": {"min_income_yield": 3.5},
    "Roth IRA": {"focus": "Capital Growth"}
}

TIER2_INDICATOR_CONFIG = {
    "Moving_Averages": {
        "sma_short_period": 20,
        "sma_medium_period": 50,
        "sma_long_period": 200,
        "weights": {"price_above_200sma": 25, "sma50_above_200sma": 20}
    },
    "RSI": {
        "period": 14,
        "oversold_threshold": 30,
        "neutral_low": 45,
        "neutral_high": 60,
        "overbought_threshold": 70
    },
    "Volatility_MACD": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "macd_bullish_cross_weight": 25
    },
    "Score_Thresholds": {
        "Strong Buy": 75,
        "Buy": 55,
        "Hold": 40
    }
}
