"""
config/portfolio.py
===================
Single source of truth for ETF universes, risk profile rules, account 
location constraints, and technical indicator parameters.
"""

# ==============================================================================
# 1. WATCHLIST & DYNAMIC SCAN POOLS
# ==============================================================================

# Core Seed Favorites List (Editable in UI session_state at runtime)
DEFAULT_FAVORITES = [
    "VFLO",  # VictoryShares Free Cash Flow ETF
    "SCHD",  # Schwab U.S. Dividend Equity ETF
    "SCYB",  # Schwab High Yield ETF
    "JPST",  # JPMorgan Ultra-Short Income ETF
    "JAAA",  # Janus Henderson AAA CLO ETF
]

# Curated Dynamic Scanning Pool (Grouped by Asset Class & Strategy)
DYNAMIC_SCAN_POOL = {
    "Core Equity & Factor": [
        "VFLO", "SCHD", "VOO", "VTI", "VYM", "DGRO", "JEPI", "JEPQ"
    ],
    "Fixed Income & Credit": [
        "JPST", "JAAA", "SCYB", "SGOV", "BIL", "MUB", "VTEB", "VCIT", "BND"
    ],
    "Growth & Sector": [
        "QQQ", "SCHG", "SMH", "VGT"
    ],
    "International & EM (Ex-China)": [
        "VEA",   # Vanguard FTSE Developed Markets
        "DIVI",  # Franklin International Core Dividend Tilt
        "EMXC",  # iShares MSCI Emerging Markets ex China
        "VEXC",  # Vanguard FTSE Emerging ex China
        "INDA",  # iShares MSCI India
        "IDV"    # iShares International Select Dividend
    ]
}

# Flattened list helper for loop scans
ALL_SCAN_TICKERS = [
    ticker for sublist in DYNAMIC_SCAN_POOL.values() for ticker in sublist
]


# ==============================================================================
# 2. TIER 1: RISK PROFILE DEFINITIONS & THRESHOLDS
# ==============================================================================

DEFAULT_RISK_RULES = {
    "Conservative": {
        "description": "Focus on capital preservation, low volatility, and steady yield.",
        "max_expense": 0.25,        # Max 0.25% Expense Ratio
        "min_yield": 3.0,          # Floor of 3.0% Annual Dividend Yield
        "max_beta": 0.60,           # Beta <= 0.60 vs S&P 500
        "min_aum_m": 500,           # Minimum $500M AUM for liquidity
        "max_volatility_3yr": 12.0  # Max 12% 3-Year Annualized Volatility
    },
    "Moderate": {
        "description": "Balanced approach combining core broad-market growth with defensive income.",
        "max_expense": 0.35,
        "min_yield": 1.5,
        "max_beta": 1.00,
        "min_aum_m": 250,
        "max_volatility_3yr": 20.0
    },
    "Aggressive": {
        "description": "Maximum long-term total return targeting secular growth and factor momentum.",
        "max_expense": 0.50,
        "min_yield": 0.0,
        "max_beta": 1.50,
        "min_aum_m": 100,
        "max_volatility_3yr": 35.0
    }
}


# ==============================================================================
# 3. TIER 1: ACCOUNT LOCATION TAX EFFICIENCY CONSTRAINTS
# ==============================================================================

ACCOUNT_LOCATION_RULES = {
    "Taxable Brokerage": {
        "tax_advantages": [
            "Capital gains taxed at preferential long-term rates (0%, 15%, 20%)",
            "Qualified dividend tax rates",
            "Unlocks Foreign Tax Credit (Form 1116) for ex-US funds (VEA, EMXC)",
            "Federal income tax exemption for Municipal Bond interest (MUB, VTEB)"
        ],
        "max_income_yield": 3.2,  # Flag funds above this yield to avoid tax drag
        "allowed_asset_types": ["Broad Equity", "Muni Bonds", "Low-Yield Foreign Equity"]
    },
    "Roth IRA": {
        "tax_advantages": [
            "100% tax-free growth and 100% tax-free qualified distributions in retirement",
            "Best location for high capital appreciation and dynamic rebalancing"
        ],
        "allowed_asset_types": ["Growth Equities", "Semiconductors/Tech", "Emerging Growth"]
    },
    "Traditional / Rollover IRA": {
        "tax_advantages": [
            "Tax-deferred growth; shelters high ordinary-income distributions until withdrawal"
        ],
        "min_income_yield": 3.8,  # Strongly prefers high ordinary income/yield
        "allowed_asset_types": ["REITs", "Corporate Bonds", "Covered Call/Options Income", "CLOs"]
    }
}


# ==============================================================================
# 4. TIER 2: TACTICAL SCORING INDICATOR CONFIGURATION
# ==============================================================================

TIER2_INDICATOR_CONFIG = {
    "Moving_Averages": {
        "sma_short_period": 20,
        "sma_medium_period": 50,
        "sma_long_period": 200,
        "weights": {
            "price_above_200sma": 25,  # Bullish trend confirmation (+25 pts)
            "sma50_above_200sma": 20   # Golden Cross (+20 pts)
        }
    },
    "RSI": {
        "period": 14,
        "oversold_threshold": 30,     # Oversold -> High Buy score (+30 pts)
        "overbought_threshold": 70,    # Overbought -> Sell/Take Profit score (-30 pts)
        "neutral_low": 40,
        "neutral_high": 60
    },
    "Volatility_MACD": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "macd_bullish_cross_weight": 25
    },
    "Score_Thresholds": {
        "Strong Buy": 75,   # Final score >= 75
        "Buy": 60,          # Final score 60-74
        "Hold": 40,         # Final score 40-59
        "Sell": 25          # Final score < 40
    }
}
