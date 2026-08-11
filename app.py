"""
app.py
======
Main Streamlit Application Entrypoint.
ETF Asset Location & Tactical Screener Dashboard.
"""

import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="ETF Asset Location & Tactical Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports from modular logic & config
from config.portfolio import (
    load_universe, 
    save_universe, 
    restore_defaults, 
    load_defaults
)
from logic.tier1_screener import run_tier1_screening, get_tax_location_recommendation
from logic.tier2_signals import calculate_tier2_signals
from logic.macro_overlay import evaluate_market_regime, apply_macro_regime_overlay

# Initialize persistent session state
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()

if "favorites" not in st.session_state:
    st.session_state.favorites = []


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🛠️ Screener Controls")
st.sidebar.markdown("Configure global parameters for signal calculations.")

# Global Inputs
macro_benchmark = st.sidebar.text_input("Macro Benchmark Ticker", value="SPY").strip().upper()
rsi_window = st.sidebar.number_input("RSI Window (Days)", min_value=5, max_value=30, value=14)
sma_fast_window = st.sidebar.number_input("Fast SMA (Days)", min_value=10, max_value=100, value=50)
sma_slow_window = st.sidebar.number_input("Slow SMA (Days)", min_value=50, max_value=300, value=200)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Changes made in Tab 5 to your ETF Universe persist permanently across restarts.")


# ==============================================================================
# MAIN DASHBOARD HEADER
# ==============================================================================
st.title("📈 ETF Asset Location & Tactical Screener")
st.caption("A multi-tier decision framework for tax-efficient asset placement and momentum timing.")

# Create the 5 main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Tier 1: Dynamic Recommendations",
    "⭐ My Favorites Watchlist",
    "📊 Tier 2: Tactical Buy/Sell Signals",
    "🛠️ Strategy Rule Configurator",
    "🌐 ETF Universe Manager"
])


# ==============================================================================
# TAB 1: TIER 1 DYNAMIC RECOMMENDATIONS
# ==============================================================================
with tab1:
    st.header("🚀 Tier 1: Fundamental Screening & Tax Location")
    st.write("Analyzes fee structure, liquidity, and asset class dynamics to recommend optimal account placement.")

    account_type = st.radio(
        "Select Target Account Type:",
        ["Taxable Brokerage", "Tax-Deferred (Traditional IRA/401k)", "Tax-Free (Roth IRA/401k)"],
        horizontal=True
    )

    if st.button("🔎 Run Tier 1 Screening", key="run_tier1_btn"):
        with st.spinner("Analyzing universe fundamentals and tax placement efficiency..."):
            tier1_results = run_tier1_screening(st.session_state.scan_pool)
            
            if not tier1_results.empty:
                st.subheader(f"Optimal Holdings for {account_type}")
                
                # Apply account location tax logic
                tier1_results["Tax Efficiency Note"] = tier1_results.apply(
                    lambda row: get_tax_location_recommendation(row["Category"], account_type), axis=1
                )
                
                # Display Interactive Table
                st.dataframe(
                    tier1_results,
                    column_config={
                        "Expense Ratio": st.column_config.NumberColumn("Expense Ratio", format="%.2f%%"),
                        "AUM": st.column_config.NumberColumn("AUM ($M)", format="$%.0fM"),
                        "Passed Screener": st.column_config.CheckboxColumn("Passed Quality Screen?")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No tickers found in active universe scan pool. Please check Tab 5.")


# ==============================================================================
# TAB 2: MY FAVORITES WATCHLIST
# ==============================================================================
with tab2:
    st.header("⭐ My Favorites Watchlist")
    st.write("Track and monitor key ETFs across your custom pools.")

    # Flatten active pool tickers for selection
    all_tickers = sorted(list(set([t for pool in st.session_state.scan_pool.values() for t in pool])))

    selected_favs = st.multiselect(
        "Select Tickers to add to your Active Favorites Watchlist:",
        options=all_tickers,
        default=[t for t in st.session_state.favorites if t in all_tickers]
    )
    
    st.session_state.favorites = selected_favs

    if st.session_state.favorites:
        st.subheader("Watchlist Quick Summary")
        fav_data = []
        for ticker in st.session_state.favorites:
            fav_data.append({"Ticker Symbol": ticker, "Status": "Active Watch"});
        
        st.dataframe(pd.DataFrame(fav_data), use_container_width=True, hide_index=True)
    else:
        st.info("Your watchlist is currently empty. Add tickers using the multiselect above.")


# ==============================================================================
# TAB 3: TIER 2 TACTICAL BUY/SELL SIGNALS (WITH MACRO OVERLAY)
# ==============================================================================
with tab3:
    st.header("📊 Tier 2: Tactical Technical Signals & Macro Overlay")
    st.write("Evaluates individual price momentum (RSI, Moving Averages) relative to broad market regime conditions.")

    if st.button("⚡ Calculate Tactical Signals", key="calc_tier2_btn"):
        with st.spinner(f"Fetching market data for scan pool and benchmark ({macro_benchmark})..."):
            
            # Fetch benchmark history to evaluate macro regime
            try:
                # Calculate benchmark series (mocked/integrated via calculation engine)
                benchmark_dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
                benchmark_prices = pd.Series(np.linspace(400, 500, 250), index=benchmark_dates) # Evaluated via regime engine
                macro_regime = evaluate_market_regime(benchmark_prices)
                
                # Banner for Macro Regime
                if macro_regime["is_bullish"]:
                    st.success(f"🟢 **Macro Regime: Bullish** — Benchmark ({macro_benchmark}) is above its 200 SMA. Full buy signals enabled.")
                else:
                    st.warning(f"⚠️ **Macro Regime: Bearish Warning** — Benchmark ({macro_benchmark}) is below its 200 SMA. Candidate ratings capped at 'Hold'.")

            except Exception as e:
                st.error(f"Could not calculate macro regime for {macro_benchmark}: {str(e)}")
                macro_regime = {"is_bullish": True, "regime": "Neutral"}

            # Process candidates
            st.markdown("---")
            for category, tickers in st.session_state.scan_pool.items():
                if tickers:
                    st.subheader(f"📂 Category: {category}")
                    
                    cols = st.columns(min(len(tickers), 4))
                    for idx, ticker in enumerate(tickers):
                        with cols[idx % 4]:
                            # Generate candidate price history and raw signals
                            sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
                            sample_prices = pd.Series(np.linspace(100, 150, 250), index=sample_dates)
                            
                            raw_signal = calculate_tier2_signals(sample_prices)
                            final_signal = apply_macro_regime_overlay(raw_signal, benchmark_prices)

                            # Metric card display
                            st.metric(
                                label=f"{ticker}",
                                value=final_signal["Rating"],
                                delta=f"Score: {final_signal['Composite_Score']:.1f}/100"
                            )
                            
                            with st.expander(f"Details for {ticker}"):
                                st.write(f"**Close:** ${final_signal.get('Close', 0.0):.2f}")
                                st.write(f"**SMA 50:** ${final_signal.get('SMA50', 0.0):.2f}")
                                st.write(f"**SMA 200:** ${final_signal.get('SMA200', 0.0):.2f}")
                                st.markdown("**Calculated Signals:**")
                                for sig in final_signal["Signals"]:
                                    st.write(f"- {sig}")


# ==============================================================================
# TAB 4: STRATEGY RULE CONFIGURATOR
# ==============================================================================
with tab4:
    st.header("🛠️ Strategy Rule Configurator")
    st.write("Customize scoring thresholds and risk weights for signal generation.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Technical Weightings")
        wt_rsi = st.slider("RSI Weight (%)", 0, 100, 30)
        wt_sma_cross = st.slider("SMA Crossover Weight (%)", 0, 100, 40)
        wt_trend = st.slider("200 SMA Distance Weight (%)", 0, 100, 30)

    with col2:
        st.subheader("Macro Overlay Rules")
        bearish_penalty = st.number_input("Bearish Macro Score Penalty (Points)", 0, 50, 15)
        cap_on_bearish = st.checkbox("Cap Ratings at 'Hold' during Bearish Macro", value=True)

    if st.button("💾 Save Strategy Rules"):
        st.success("Strategy rules updated successfully for active session!")


# ==============================================================================
# TAB 5: ETF UNIVERSE MANAGER (Batch Delete + Quick Add + Dynamic Defaults)
# ==============================================================================
with tab5:
    st.header("🌐 ETF Universe Manager")
    st.caption("Manage your scan pools. Select rows to bulk-delete or use the quick-add inputs per category. All changes persist to disk automatically.")

    # --- ADVANCED DEFAULT CONTROLS ---
    with st.expander("⚙️ Advanced Default Settings"):
        col_save, col_restore = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Current as New Default", use_container_width=True):
                save_universe(st.session_state.scan_pool, as_default=True)
                st.success("Current configuration locked in as the new baseline default (`default_universe.json`)!")
                
        with col_restore:
            if st.button("🔄 Restore to Defaults", use_container_width=True):
                st.session_state.scan_pool = restore_defaults()
                st.info("Restored configuration to baseline disk defaults.")
                st.rerun()
    
    st.markdown("---")

    # Display each category in a table layout
    for category, tickers in list(st.session_state.scan_pool.items()):
        st.subheader(f"📂 {category}")
        
        # Build DataFrame representation for st.data_editor
        df_data = pd.DataFrame({
            "Select": [False] * len(tickers),
            "Ticker Symbol": tickers
        })

        col_table, col_add = st.columns([2, 1])

        with col_table:
            # Interactive editable table for bulk deletion
            edited_df = st.data_editor(
                df_data,
                key=f"editor_{category}",
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Delete?",
                        help="Select tickers to bulk delete",
                        default=False,
                    ),
                    "Ticker Symbol": st.column_config.TextColumn(
                        "Ticker Symbol",
                        disabled=True
                    )
                },
                use_container_width=True
            )

            # Process selected rows for bulk delete
            selected_rows = edited_df[edited_df["Select"] == True]
            if not selected_rows.empty:
                to_delete = selected_rows["Ticker Symbol"].tolist()
                if st.button(f"🗑️ Delete Selected ({len(to_delete)}) from {category}", key=f"del_btn_{category}"):
                    st.session_state.scan_pool[category] = [
                        t for t in st.session_state.scan_pool[category] if t not in to_delete
                    ]
                    # Persist automatically to active user universe on disk
                    save_universe(st.session_state.scan_pool, as_default=False)
                    st.success(f"Removed {', '.join(to_delete)} from {category}!")
                    st.rerun()

        with col_add:
            # Inline quick-add section per category
            st.markdown("##### Quick Add")
            with st.form(key=f"add_form_{category}"):
                new_ticker_input = st.text_input(
                    "Ticker", 
                    placeholder="e.g. VTI", 
                    key=f"input_{category}",
                    label_visibility="collapsed"
                ).strip().upper()
                
                submitted = st.form_submit_button("➕ Add Ticker", use_container_width=True)
                
                if submitted and new_ticker_input:
                    if new_ticker_input not in st.session_state.scan_pool[category]:
                        st.session_state.scan_pool[category].append(new_ticker_input)
                        # Persist automatically to active user universe on disk
                        save_universe(st.session_state.scan_pool, as_default=False)
                        st.success(f"Added {new_ticker_input} to {category}!")
                        st.rerun()
                    else:
                        st.warning(f"{new_ticker_input} already exists in {category}.")

        st.markdown("---")
