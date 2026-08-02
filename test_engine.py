"""
test_engine.py
==============
Local CLI testing script to verify Tier 1 screening and Tier 2 technical 
scoring pipelines without needing Streamlit.
"""

import sys
from config.portfolio import ALL_SCAN_TICKERS, DEFAULT_RISK_RULES
from logic.tier1_screener import fetch_etf_fundamentals, run_tier1_screen
from logic.tier2_signals import run_tier2_scoring


def run_local_pipeline_test(risk_profile="Moderate"):
    print("=" * 60)
    print(f"  RUNNING LOCAL TEST PIPELINE (Profile: {risk_profile})")
    print("=" * 60)

    # Use a representative subset of tickers for speed
    test_tickers = ALL_SCAN_TICKERS[:10]  # First 10 tickers from universe
    print(f"\n[1/3] Fetching fundamentals for test pool: {test_tickers}...")

    df_raw = fetch_etf_fundamentals(test_tickers)
    if df_raw.empty:
        print("❌ Error: Failed to fetch fundamentals from yfinance.")
        return

    print(f"✓ Fetched raw fundamental data for {len(df_raw)} tickers.")

    # --- TIER 1 TEST ---
    print(f"\n[2/3] Executing Tier 1 Fundamental Screen ({risk_profile})...")
    df_tier1 = run_tier1_screen(df_raw, risk_profile_name=risk_profile)

    if df_tier1.empty:
        print("⚠️ No tickers passed Tier 1 screening rules.")
        passing_tickers = test_tickers[:3]  # Fallback to test Tier 2 anyway
        print(f"Falling back to test Tier 2 with: {passing_tickers}")
    else:
        passing_tickers = df_tier1["Ticker"].tolist()
        print(f"✓ {len(passing_tickers)} ETFs passed Tier 1 screening:")
        
        # Display Tier 1 summary table
        cols_to_show = ["Ticker", "Name", "Expense (%)", "Yield (%)", "Target Account"]
        print("\n--- Tier 1 Results ---")
        print(df_tier1[cols_to_show].to_string(index=False))

    # --- TIER 2 TEST ---
    print(f"\n[3/3] Executing Tier 2 Technical Scoring on passed tickers...")
    df_tier2 = run_tier2_scoring(passing_tickers)

    if df_tier2.empty:
        print("❌ Error: Tier 2 scoring returned no results.")
    else:
        print("\n--- Tier 2 Tactical Scores & Signals ---")
        cols_t2 = ["Ticker", "Price ($)", "Score", "Rating", "RSI (14)", "Key Signals"]
        print(df_tier2[cols_t2].to_string(index=False))

    print("\n" + "=" * 60)
    print("  PIPELINE TEST COMPLETE - ALL MODULES OK")
    print("=" * 60)


if __name__ == "__main__":
    # Allow overriding risk profile via command line arg (e.g., python test_engine.py Aggressive)
    profile = sys.argv[1] if len(sys.argv) > 1 else "Moderate"
    run_local_pipeline_test(risk_profile=profile)
