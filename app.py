"""
app.py
======
Main Streamlit dashboard interface for the ETF Portfolio Management System.
Includes Tier 1 Fundamental Screening, Watchlist Management, Tier 2 Signals,
Strategy Rule Configurator, and Live ETF Universe Management.
"""

import sys
import os

# Guarantee project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="ETF Tactical & Fundamental Screener",
    page_icon="📈",
    layout="wide"
)

# Safe Imports with Exception Handling
try:
    from config.portfolio import (
        DYNAMIC_SCAN_POOL,
        DEFAULT_RISK_RULES,
        DEFAULT_FAVORITES,
        TIER2_INDICATOR_CONFIG
    )
    from logic.tier1_screener import fetch_etf_fundamentals, run_tier1_screen, map_account_location
    from logic.tier2_signals import fetch_historical_prices, calculate_tier2_signals
except Exception as e:
    st.error(f"❌ Detailed Module Load Error: {e}")
    st.write("---")
    st.caption("Common causes:")
    st.caption("1. Relative imports inside `logic/tier1_screener.py` (e.g., `from ..config import ...`).")
    st.caption("2. A missing dependency in `requirements.txt`.")
    st.stop()

# Initialize Session States
if "risk_rules" not in st.session_state:
    st.session_state.risk_rules = DEFAULT_RISK_RULES.copy()

if "favorites" not in st.session_state:
    st.session_state.favorites = DEFAULT_FAVORITES.copy()

if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = DYNAMIC_SCAN_POOL.copy()

# Header & Sidebar Configuration
st.title("📈 ETF Asset Location & Tactical Screener")

with st.sidebar:
    st.header("⚙️ Profile & Settings")
    
    selected_profile = st.selectbox(
        "Active Risk Profile:",
        options=list(st.session_state.risk_rules.keys()),
        index=1
    )
    
    current_rules = st.session_state.risk_rules[selected_profile]
    
    st.subheader("Active Profile Thresholds")
    st.write(f"• **Max Expense:** `{current_rules['max_expense']}%`")
    st.write(f"• **Min Yield:** `{current_rules['min_yield']}%`")
    st.write(f"• **Max Beta:** `{current_rules['max_beta']}`")
    st.write(f"• **Min AUM:** `${current_rules['min_aum_m']}M`")
    st.write(f"• **Max 3Yr Vol:** `{current_rules['max_volatility_3yr']}%`")
    
    st.markdown("---")
    st.metric("Saved Favorites Count", len(st.session_state.favorites))

# Dashboard Navigation
tab_tier1, tab_watchlist, tab_tier2, tab_config, tab_universe = st.tabs([
    "🚀 Tier 1: Dynamic Recommendations",
    "⭐ My Favorites Watchlist",
    "📊 Tier 2: Tactical Buy/Sell Signals",
    "🛠️ Strategy Rule Configurator",
    "🌐 ETF Universe Manager"
])

# Flatten all active tickers across sub-pools
all_scan_tickers = list(set([t for pool in st.session_state.scan_pool.values() for t in pool]))

# ==============================================================================
# TAB 1: TIER 1 FUNDAMENTAL SCREENER
# ==============================================================================
with tab_tier1:
    st.header("🚀 Tier 1 Fundamental Screening")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        pool_selection = st.selectbox(
            "Scan Sub-Pool:",
            options=["All Assets"] + list(st.session_state.scan_pool.keys())
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Refresh Market Scan", use_container_width=True)

    # Determine target list
    if pool_selection == "All Assets":
        target_tickers = all_scan_tickers
    else:
        target_tickers = st.session_state.scan_pool.get(pool_selection, [])

    with st.spinner("Fetching fundamentals & running screening rules..."):
        df_raw = fetch_etf_fundamentals(target_tickers)
        df_screened = run_tier1_screen(df_raw, current_rules)

    if not df_screened.empty:
        df_screened["Account Location"] = df_screened.apply(map_account_location, axis=1)

    st.markdown(f"**Screening Results:** Passed **{len(df_screened)}** of **{len(target_tickers)}** evaluated funds.")

    tab_taxable, tab_roth, tab_trad = st.tabs([
        "🏦 Taxable Brokerage", 
        "📈 Roth IRA", 
        "🛡️ Traditional / Rollover IRA"
    ])

    def render_fund_table(df_subset, account_type):
        if df_subset.empty:
            st.info(f"No ETFs currently match the {account_type} criteria under this profile.")
            return

        for idx, row in df_subset.iterrows():
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.5, 3, 2, 2, 2])
                
                ticker = row["Ticker"]
                is_fav = ticker in st.session_state.favorites
                
                with c1:
                    st.subheader(ticker)
                    btn_label = "⭐ Saved" if is_fav else "⭐ Fav"
                    if st.button(btn_label, key=f"fav_btn_{account_type}_{ticker}"):
                        if is_fav:
                            st.session_state.favorites.remove(ticker)
                        else:
                            st.session_state.favorites.append(ticker)
                        st.rerun()

                with c2:
                    st.write(f"**{row.get('Name', ticker)}**")
                    st.caption(f"Category: {row.get('Category', 'N/A')}")

                with c3:
                    st.write(f"**Expense:** `{row['Expense_Ratio']:.2f}%`")
                    st.write(f"**Yield:** `{row['Dividend_Yield']:.2f}%`")

                with c4:
                    st.write(f"**Beta:** `{row['Beta']:.2f}`")
                    st.write(f"**AUM:** `${row['AUM_M']:.0f}M`")

                with c5:
                    st.write(f"**3Yr Vol:** `{row['Volatility_3Yr']:.1f}%`")

                st.markdown("---")

    with tab_taxable:
        if not df_screened.empty:
            render_fund_table(df_screened[df_screened["Account Location"] == "Taxable Brokerage"], "Taxable")
        else:
            st.info("No funds passed Tier 1 criteria.")

    with tab_roth:
        if not df_screened.empty:
            render_fund_table(df_screened[df_screened["Account Location"] == "Roth IRA"], "Roth")
        else:
            st.info("No funds passed Tier 1 criteria.")

    with tab_trad:
        if not df_screened.empty:
            render_fund_table(df_screened[df_screened["Account Location"] == "Traditional IRA"], "Traditional")
        else:
            st.info("No funds passed Tier 1 criteria.")

# ==============================================================================
# TAB 2: MY FAVORITES WATCHLIST
# ==============================================================================
with tab_watchlist:
    st.header("⭐ My Favorites Watchlist")
    
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        new_fav = st.text_input("Quick Add Ticker Symbol:", key="add_fav_input").strip().upper()
    with col_w2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add to Watchlist", use_container_width=True):
            if new_fav and new_fav not in st.session_state.favorites:
                st.session_state.favorites.append(new_fav)
                st.success(f"Added {new_fav} to Watchlist!")
                st.rerun()

    if st.session_state.favorites:
        fav_df = fetch_etf_fundamentals(st.session_state.favorites)
        
        for idx, row in fav_df.iterrows():
            ticker = row["Ticker"]
            with st.container():
                wc1, wc2, wc3, wc4 = st.columns([1, 3, 3, 1])
                with wc1:
                    st.subheader(ticker)
                with wc2:
                    st.write(f"**{row.get('Name', ticker)}**")
                with wc3:
                    st.write(f"Expense: `{row['Expense_Ratio']:.2f}%` | Yield: `{row['Dividend_Yield']:.2f}%` | Beta: `{row['Beta']:.2f}`")
                with wc4:
                    if st.button("❌ UnFav", key=f"unfav_{ticker}"):
                        st.session_state.favorites.remove(ticker)
                        st.toast(f"Removed {ticker} from watchlist.")
                        st.rerun()
            st.markdown("---")
    else:
        st.info("Your watchlist is currently empty. Star ETFs from Tier 1 or enter a symbol above!")

# ==============================================================================
# TAB 3: TIER 2 TACTICAL SIGNALS
# ==============================================================================
with tab_tier2:
    st.header("📊 Tier 2 Tactical Buy/Sell Signals")
    st.caption("Technical analysis and momentum scoring for your watchlisted assets.")

    if not st.session_state.favorites:
        st.warning("Please add ETFs to your favorites watchlist to generate tactical signals.")
    else:
        with st.spinner("Fetching price history & calculating technical indicators..."):
            hist_data = fetch_historical_prices(st.session_state.favorites)
            
            if not hist_data.empty:
                signal_results = []
                for t in st.session_state.favorites:
                    if t in hist_data.columns:
                        res = calculate_tier2_signals(hist_data[t], TIER2_INDICATOR_CONFIG)
                        res["Ticker"] = t
                        signal_results.append(res)
                
                sig_df = pd.DataFrame(signal_results)
                
                for idx, row in sig_df.iterrows():
                    with st.container():
                        sc1, sc2, sc3 = st.columns([1.5, 2.5, 4])
                        
                        with sc1:
                            st.subheader(row["Ticker"])
                            rating = row["Rating"]
                            if rating == "Strong Buy":
                                st.success(f"🟢 {rating}")
                            elif rating == "Buy":
                                st.info(f"🔵 {rating}")
                            elif rating == "Hold":
                                st.warning(f"🟡 {rating}")
                            else:
                                st.error(f"🔴 {rating}")
                            st.metric("Composite Score", f"{row['Composite_Score']:.0f} / 100")

                        with sc2:
                            st.write(f"**Latest Price:** `${row['Close']:.2f}`")
                            st.write(f"**RSI (14):** `{row['RSI']:.1f}`")
                            st.write(f"**200-day SMA:** `${row['SMA200']:.2f}`")
                            st.write(f"**50-day SMA:** `${row['SMA50']:.2f}`")

                        with sc3:
                            st.write("**Tactical Signals:**")
                            for sig in row["Signals"]:
                                st.write(f"• {sig}")

                    st.markdown("---")

# ==============================================================================
# TAB 4: STRATEGY RULE CONFIGURATOR
# ==============================================================================
with tab_config:
    st.header("🛠️ Strategy Rule Configurator")
    st.caption("Customize the exact numerical thresholds used by the Tier 1 screening engine.")

    selected_edit_profile = st.selectbox(
        "Select Profile to Edit:",
        list(st.session_state.risk_rules.keys()),
        index=0
    )

    p_rules = st.session_state.risk_rules[selected_edit_profile]

    st.subheader(f"Editing: {selected_edit_profile} Profile")
    st.info(p_rules["description"])

    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### Fundamental Limits")
        new_max_exp = st.slider(
            "Max Expense Ratio (%)",
            min_value=0.01, max_value=1.50, value=float(p_rules["max_expense"]), step=0.01,
            key=f"exp_{selected_edit_profile}"
        )
        new_min_yield = st.slider(
            "Min Dividend Yield (%) — Set to 0.0% for Broad Screening",
            min_value=0.0, max_value=6.0, value=float(p_rules["min_yield"]), step=0.1,
            key=f"yield_{selected_edit_profile}"
        )
        new_min_aum = st.number_input(
            "Min AUM ($ Millions)",
            min_value=0, max_value=10000, value=int(p_rules["min_aum_m"]), step=50,
            key=f"aum_{selected_edit_profile}"
        )

    with col_c2:
        st.markdown("#### Risk & Volatility Limits")
        new_max_beta = st.slider(
            "Max Beta (vs S&P 500)",
            min_value=0.10, max_value=2.50, value=float(p_rules["max_beta"]), step=0.05,
            key=f"beta_{selected_edit_profile}"
        )
        new_max_vol = st.slider(
            "Max 3-Year Annualized Volatility (%)",
            min_value=5.0, max_value=60.0, value=float(p_rules["max_volatility_3yr"]), step=0.5,
            key=f"vol_{selected_edit_profile}"
        )

    st.markdown("---")
    
    col_save, col_reset = st.columns([2, 2])
    with col_save:
        if st.button(f"💾 Save & Apply {selected_edit_profile} Rule Changes", use_container_width=True):
            st.session_state.risk_rules[selected_edit_profile].update({
                "max_expense": new_max_exp,
                "min_yield": new_min_yield,
                "min_aum_m": new_min_aum,
                "max_beta": new_max_beta,
                "max_volatility_3yr": new_max_vol
            })
            st.success(f"Successfully updated rule parameters for {selected_edit_profile}!")
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset All Profiles to Factory Defaults", use_container_width=True):
            st.session_state.risk_rules = DEFAULT_RISK_RULES.copy()
            st.toast("Reset all risk profile rules to default values.")
            st.rerun()

# ==============================================================================
# TAB 5: ETF UNIVERSE MANAGER
# ==============================================================================
with tab_universe:
    st.header("🌐 ETF Universe Manager")
    st.caption("Add, remove, or organize tickers across sub-pools dynamically.")

    all_current_tickers = list(set([t for pool in st.session_state.scan_pool.values() for t in pool]))
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.metric("Total Universe Tickers", len(all_current_tickers))
    with col_u2:
        st.metric("Active Sub-Categories", len(st.session_state.scan_pool))

    st.markdown("---")

    st.subheader("➕ Add New Ticker to Universe")
    
    col_add1, col_add2, col_add3 = st.columns([2, 2, 1])
    
    with col_add1:
        new_ticker = st.text_input("Ticker Symbol (e.g., USMV):", key="new_univ_ticker").strip().upper()
    
    with col_add2:
        target_category = st.selectbox(
            "Select Sub-Pool Category:", 
            options=list(st.session_state.scan_pool.keys()),
            key="target_category"
        )
        
    with col_add3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add Ticker", use_container_width=True):
            if new_ticker:
                if new_ticker not in st.session_state.scan_pool[target_category]:
                    st.session_state.scan_pool[target_category].append(new_ticker)
                    st.success(f"Added **{new_ticker}** to **{target_category}**!")
                    st.rerun()
                else:
                    st.warning(f"**{new_ticker}** is already in **{target_category}**.")
            else:
                st.error("Please enter a valid ticker symbol.")

    st.markdown("---")

    st.subheader("🗂️ Manage Existing Sub-Pools")
    
    for category, ticker_list in st.session_state.scan_pool.items():
        with st.expander(f"📁 {category} ({len(ticker_list)} Tickers)", expanded=True):
            st.write("Current Tickers:", ", ".join([f"`{t}`" for t in ticker_list]))
            
            ticker_to_remove = st.selectbox(
                f"Remove Ticker from {category}:",
                options=["-- Select Ticker to Remove --"] + ticker_list,
                key=f"remove_{category}"
            )
            
            if ticker_to_remove != "-- Select Ticker to Remove --":
                if st.button(f"❌ Remove {ticker_to_remove} from {category}", key=f"btn_rem_{category}_{ticker_to_remove}"):
                    st.session_state.scan_pool[category].remove(ticker_to_remove)
                    st.toast(f"Removed {ticker_to_remove} from {category}")
                    st.rerun()

    st.markdown("---")

    with st.expander("➕ Create New Sub-Pool Category"):
        new_cat_name = st.text_input("New Category Name (e.g., 'Real Estate & Commodities'):").strip()
        if st.button("Create Category"):
            if new_cat_name and new_cat_name not in st.session_state.scan_pool:
                st.session_state.scan_pool[new_cat_name] = []
                st.success(f"Created new category: **{new_cat_name}**")
                st.rerun()
            elif new_cat_name in st.session_state.scan_pool:
                st.warning("Category already exists.")
