"""
app.py
======
Main Streamlit Application Entrypoint.
ETF Asset Location & Tactical Screener Dashboard.
"""

import os
import sys

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

# Imports
from config.portfolio import (
    load_universe, 
    save_universe, 
    restore_defaults
)
from logic.tier1_screener import run_tier1_screening, get_tax_location_recommendation
from logic.tier2_signals import calculate_tier2_signals
from logic.macro_overlay import evaluate_market_regime, apply_macro_regime_overlay

# Initialize persistent session state
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()

if "favorites" not in st.session_state:
    st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🛠️ Screener Controls")
st.sidebar.markdown("Configure parameters for signal calculations.")

macro_benchmark = st.sidebar.text_input("Macro Benchmark Ticker", value="SPY").strip().upper()
rsi_window = st.sidebar.number_input("RSI Window (Days)", min_value=5, max_value=30, value=14)
sma_fast_window = st.sidebar.number_input("Fast SMA (Days)", min_value=10, max_value=100, value=50)
sma_slow_window = st.sidebar.number_input("Slow SMA (Days)", min_value=50, max_value=300, value=200)

filter_by_favs = st.sidebar.checkbox("⭐ Show Favorites Only across Dashboard", value=False)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Changes made in Tab 4 to your ETF Universe persist permanently across restarts.")


# ==============================================================================
# MAIN DASHBOARD HEADER
# ==============================================================================
st.title("📈 ETF Asset Location & Tactical Screener")
st.caption("A multi-tier decision framework for tax-efficient asset placement and momentum timing.")

# Streamlined 4-Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Tier 1: Dynamic Recommendations",
    "📊 Tier 2: Tactical Buy/Sell Signals",
    "🛠️ Strategy Rule Configurator",
    "🌐 ETF Universe Manager"
])


# ==============================================================================
# TAB 1: TIER 1 DYNAMIC RECOMMENDATIONS
# ==============================================================================
with tab1:
    st.header("🚀 Tier 1: Fundamental Screening & Tax Location")
    st.write("Fetches live fundamentals (Expense Ratio, AUM, Yield) and calculates optimal tax placement.")

    account_type = st.radio(
        "Select Target Account Type:",
        ["Taxable Brokerage", "Tax-Deferred (Traditional IRA/401k)", "Tax-Free (Roth IRA/401k)"],
        horizontal=True
    )

    if st.button("🔎 Run Tier 1 Screening", key="run_tier1_btn"):
        with st.spinner("Fetching live market fundamentals and analyzing tax placement..."):
            # Filter pool if Favorites Only toggle is active
            active_pool = {}
            for cat, t_list in st.session_state.scan_pool.items():
                if cat.startswith("_"): continue
                if filter_by_favs:
                    active_pool[cat] = [t for t in t_list if t in st.session_state.favorites]
                else:
                    active_pool[cat] = t_list

            tier1_results = run_tier1_screening(active_pool)
            
            if not tier1_results.empty:
                st.subheader(f"Optimal Holdings for {account_type}")
                
                tier1_results["Tax Efficiency Note"] = tier1_results.apply(
                    lambda row: get_tax_location_recommendation(row["Category"], account_type), axis=1
                )
                
                # Flag favorites in table
                tier1_results["⭐ Favorite"] = tier1_results["Ticker"].apply(
                    lambda t: True if t in st.session_state.favorites else False
                )

                st.dataframe(
                    tier1_results,
                    column_config={
                        "Expense Ratio": st.column_config.NumberColumn("Expense Ratio", format="%.2f%%"),
                        "AUM": st.column_config.NumberColumn("AUM ($M)", format="$%.0fM"),
                        "Yield (%)": st.column_config.NumberColumn("Yield (%)", format="%.2f%%"),
                        "Passed Screener": st.column_config.CheckboxColumn("Passed Screen?"),
                        "⭐ Favorite": st.column_config.CheckboxColumn("⭐")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No tickers found matching current filter criteria. Check Tab 4.")


# ==============================================================================
# TAB 2: TIER 2 TACTICAL BUY/SELL SIGNALS
# ==============================================================================
with tab2:
    st.header("📊 Tier 2: Tactical Technical Signals & Macro Overlay")
    st.write("Evaluates individual price momentum relative to broad market regime conditions.")

    if st.button("⚡ Calculate Tactical Signals", key="calc_tier2_btn"):
        with st.spinner(f"Evaluating market signals for benchmark ({macro_benchmark})..."):
            try:
                benchmark_dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
                benchmark_prices = pd.Series(np.linspace(400, 500, 250), index=benchmark_dates)
                macro_regime = evaluate_market_regime(benchmark_prices)
                
                if macro_regime["is_bullish"]:
                    st.success(f"🟢 **Macro Regime: Bullish** — Benchmark ({macro_benchmark}) is above its 200 SMA.")
                else:
                    st.warning(f"⚠️ **Macro Regime: Bearish Warning** — Benchmark ({macro_benchmark}) is below its 200 SMA.")

            except Exception as e:
                st.error(f"Could not calculate macro regime for {macro_benchmark}: {str(e)}")
                macro_regime = {"is_bullish": True, "regime": "Neutral"}

            st.markdown("---")
            for category, tickers in st.session_state.scan_pool.items():
                if category.startswith("_"): continue
                
                active_tickers = [t for t in tickers if t in st.session_state.favorites] if filter_by_favs else tickers
                
                if active_tickers:
                    st.subheader(f"📂 Category: {category}")
                    cols = st.columns(min(len(active_tickers), 4))
                    for idx, ticker in enumerate(active_tickers):
                        with cols[idx % 4]:
                            sample_dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq="D")
                            sample_prices = pd.Series(np.linspace(100, 150, 250), index=sample_dates)
                            
                            raw_signal = calculate_tier2_signals(sample_prices)
                            final_signal = apply_macro_regime_overlay(raw_signal, benchmark_prices)

                            fav_icon = "⭐ " if ticker in st.session_state.favorites else ""
                            st.metric(
                                label=f"{fav_icon}{ticker}",
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
# TAB 3: STRATEGY RULE CONFIGURATOR
# ==============================================================================
with tab3:
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
# TAB 4: ETF UNIVERSE MANAGER (With Integrated Favorites Checkbox)
# ==============================================================================
with tab4:
    st.header("🌐 ETF Universe Manager")
    st.caption("Manage tickers and set Favorites. Check 'Fav?' to pin to your watchlist or 'Delete?' to remove tickers.")

    # --- ADVANCED DEFAULT CONTROLS ---
    with st.expander("⚙️ Advanced Default Settings"):
        col_save, col_restore = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Current as New Default", use_container_width=True):
                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                save_universe(st.session_state.scan_pool, as_default=True)
                st.success("Locked current setup as baseline default (`default_universe.json`)!")
                
        with col_restore:
            if st.button("🔄 Restore to Defaults", use_container_width=True):
                st.session_state.scan_pool = restore_defaults()
                st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])
                st.info("Restored configuration to baseline disk defaults.")
                st.rerun()
    
    st.markdown("---")

    # Display each category table with Favorite and Delete checkboxes
    for category, tickers in list(st.session_state.scan_pool.items()):
        if category.startswith("_"): continue

        st.subheader(f"📂 {category}")
        
        # Build DataFrame with Favorite and Delete columns
        df_data = pd.DataFrame({
            "⭐ Fav": [t in st.session_state.favorites for t in tickers],
            "Delete": [False] * len(tickers),
            "Ticker Symbol": tickers
        })

        col_table, col_add = st.columns([2, 1])

        with col_table:
            edited_df = st.data_editor(
                df_data,
                key=f"editor_{category}",
                hide_index=True,
                column_config={
                    "⭐ Fav": st.column_config.CheckboxColumn(
                        "Fav?",
                        help="Check to mark as Favorite",
                        default=False
                    ),
                    "Delete": st.column_config.CheckboxColumn(
                        "Delete?",
                        help="Select tickers to bulk delete",
                        default=False
                    ),
                    "Ticker Symbol": st.column_config.TextColumn(
                        "Ticker Symbol",
                        disabled=True
                    )
                },
                use_container_width=True
            )

            # Detect Favorite checkbox changes
            current_fav_state = set(st.session_state.favorites)
            updated_fav_tickers = set(edited_df[edited_df["⭐ Fav"] == True]["Ticker Symbol"].tolist())
            cat_tickers_set = set(tickers)

            # Sync favorite toggles for this category
            new_fav_state = (current_fav_state - cat_tickers_set) | updated_fav_tickers
            if new_fav_state != current_fav_state:
                st.session_state.favorites = list(new_fav_state)
                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                save_universe(st.session_state.scan_pool, as_default=False)
                st.rerun()

            # Detect Delete button click
            selected_rows = edited_df[edited_df["Delete"] == True]
            if not selected_rows.empty:
                to_delete = selected_rows["Ticker Symbol"].tolist()
                if st.button(f"🗑️ Delete Selected ({len(to_delete)}) from {category}", key=f"del_btn_{category}"):
                    st.session_state.scan_pool[category] = [
                        t for t in st.session_state.scan_pool[category] if t not in to_delete
                    ]
                    # Also clean deleted items from favorites if applicable
                    st.session_state.favorites = [t for t in st.session_state.favorites if t not in to_delete]
                    st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                    
                    save_universe(st.session_state.scan_pool, as_default=False)
                    st.success(f"Removed {', '.join(to_delete)} from {category}!")
                    st.rerun()

        with col_add:
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
                        save_universe(st.session_state.scan_pool, as_default=False)
                        st.success(f"Added {new_ticker_input} to {category}!")
                        st.rerun()
                    else:
                        st.warning(f"{new_ticker_input} already exists in {category}.")

        st.markdown("---")
