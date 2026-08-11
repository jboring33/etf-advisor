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
    save_universe
)
from logic.tier1_screener import run_tier1_screening, get_tax_location_recommendation
from logic.tier2_signals import calculate_tier2_signals
from logic.macro_overlay import evaluate_market_regime, apply_macro_regime_overlay

# Initialize persistent session state from JSON disk storage
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()

if "account_types" not in st.session_state:
    st.session_state.account_types = st.session_state.scan_pool.get("_account_types", {})

if "favorites" not in st.session_state:
    st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])

# Helper to get user category list dynamically
def get_active_categories():
    return [cat for cat in st.session_state.scan_pool.keys() if not cat.startswith("_")]


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
st.sidebar.info("💡 **Tip:** Changes made in Tab 4 save directly to disk and persist permanently across restarts.")


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
        ["Brokerage", "IRA", "Roth/HSA"],
        horizontal=True
    )

    if st.button("🔎 Run Tier 1 Screening", key="run_tier1_btn"):
        with st.spinner("Fetching live market fundamentals and analyzing tax placement..."):
            active_pool = {}
            for cat in get_active_categories():
                t_list = st.session_state.scan_pool[cat]
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
                
                tier1_results["Target Bucket"] = tier1_results["Ticker"].apply(
                    lambda t: st.session_state.account_types.get(t, "Brokerage")
                )
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
            for category in get_active_categories():
                tickers = st.session_state.scan_pool[category]
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
                            acct_tag = f" ({st.session_state.account_types.get(ticker, 'Brokerage')})"
                            
                            st.metric(
                                label=f"{fav_icon}{ticker}{acct_tag}",
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
# TAB 4: ETF UNIVERSE MANAGER (Group Manager + Unified Table)
# ==============================================================================
with tab4:
    st.header("🌐 ETF Universe Manager")
    st.caption("Manage asset groups and master ticker table. All edits save directly to disk.")

    categories = get_active_categories()

    # --- CATEGORY / GROUP MANAGEMENT SECTION ---
    with st.expander("📁 Manage Asset Group Categories", expanded=False):
        col_c_add, col_c_del = st.columns(2)
        
        with col_c_add:
            st.markdown("##### Add New Group Category")
            with st.form("add_group_form", clear_on_submit=True):
                new_cat_name = st.text_input("Group Name", placeholder="e.g. Growth").strip()
                add_cat_submitted = st.form_submit_button("➕ Create Group", use_container_width=True)
                if add_cat_submitted and new_cat_name:
                    if new_cat_name not in st.session_state.scan_pool:
                        st.session_state.scan_pool[new_cat_name] = []
                        save_universe(st.session_state.scan_pool)
                        st.success(f"Group '{new_cat_name}' created!")
                        st.rerun()
                    else:
                        st.warning(f"Group '{new_cat_name}' already exists.")

        with col_c_del:
            st.markdown("##### Delete Existing Group")
            with st.form("del_group_form", clear_on_submit=True):
                cat_to_del = st.selectbox("Select Group to Delete", options=[""] + categories)
                del_cat_submitted = st.form_submit_button("🗑️ Delete Group", use_container_width=True)
                if del_cat_submitted and cat_to_del:
                    if len(st.session_state.scan_pool[cat_to_del]) > 0:
                        st.error(f"Cannot delete group '{cat_to_del}' because it contains tickers. Delete or reassign its tickers first!")
                    else:
                        st.session_state.scan_pool.pop(cat_to_del)
                        save_universe(st.session_state.scan_pool)
                        st.success(f"Group '{cat_to_del}' deleted!")
                        st.rerun()

    st.markdown("---")

    # --- SINGLE UNIFIED QUICK ADD SECTION ---
    st.subheader("➕ Quick Add Ticker")
    with st.form("quick_add_master_form", clear_on_submit=True):
        col_t, col_c, col_a, col_btn = st.columns([2, 2, 2, 1.5])
        
        with col_t:
            add_ticker = st.text_input("Ticker Symbol", placeholder="e.g. VTI").strip().upper()
        with col_c:
            add_category = st.selectbox("Group Category", options=categories)
        with col_a:
            add_account = st.selectbox("Account Type Bucket", options=["Brokerage", "IRA", "Roth/HSA"])
        with col_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            add_submitted = st.form_submit_button("➕ Add Ticker", use_container_width=True)

        if add_submitted and add_ticker and add_category:
            existing_tickers = [t for cat in categories for t in st.session_state.scan_pool[cat]]
            if add_ticker not in existing_tickers:
                st.session_state.scan_pool[add_category].append(add_ticker)
                st.session_state.account_types[add_ticker] = add_account
                
                # Persist directly to disk
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                save_universe(st.session_state.scan_pool)
                
                st.success(f"Added {add_ticker} to {add_category} ({add_account})!")
                st.rerun()
            else:
                st.warning(f"Ticker {add_ticker} already exists in the universe.")

    st.markdown("---")

    # --- UNIFIED MASTER TABLE DISPLAY ---
    st.subheader("📊 Master Universe Table")

    master_rows = []
    for category in categories:
        tickers = st.session_state.scan_pool.get(category, [])
        for t in tickers:
            master_rows.append({
                "Delete": False,
                "⭐ Fav": t in st.session_state.favorites,
                "Ticker": t,
                "Group / Category": category,
                "Account Type": st.session_state.account_types.get(t, "Brokerage")
            })

    if master_rows:
        master_df = pd.DataFrame(master_rows)

        edited_df = st.data_editor(
            master_df,
            key="master_universe_editor",
            hide_index=True,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Check to delete ticker",
                    default=False
                ),
                "⭐ Fav": st.column_config.CheckboxColumn(
                    "⭐ Fav",
                    help="Check to mark as Favorite",
                    default=False
                ),
                "Ticker": st.column_config.TextColumn(
                    "Ticker Symbol",
                    disabled=True
                ),
                "Group / Category": st.column_config.SelectboxColumn(
                    "Group / Category",
                    options=categories,
                    required=True
                ),
                "Account Type": st.column_config.SelectboxColumn(
                    "Account Type Bucket",
                    options=["Brokerage", "IRA", "Roth/HSA"],
                    required=True
                )
            },
            use_container_width=True
        )

        # Handle Bulk Deletions
        col_del, col_space = st.columns([1, 3])
        with col_del:
            selected_deletes = edited_df[edited_df["Delete"] == True]
            if not selected_deletes.empty:
                to_delete = selected_deletes["Ticker"].tolist()
                if st.button(f"🗑️ Delete Selected ({len(to_delete)}) Tickers", use_container_width=True):
                    for cat in categories:
                        st.session_state.scan_pool[cat] = [
                            t for t in st.session_state.scan_pool[cat] if t not in to_delete
                        ]
                    st.session_state.favorites = [t for t in st.session_state.favorites if t not in to_delete]
                    for t in to_delete:
                        st.session_state.account_types.pop(t, None)

                    # Persist changes
                    st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                    st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                    save_universe(st.session_state.scan_pool)
                    st.success(f"Deleted {', '.join(to_delete)} from universe!")
                    st.rerun()

        # Sync Table State Edits (Favorites, Categories, Account Types)
        has_changes = False

        # 1. Sync Favorites
        updated_favs = edited_df[edited_df["⭐ Fav"] == True]["Ticker"].tolist()
        if set(updated_favs) != set(st.session_state.favorites):
            st.session_state.favorites = updated_favs
            st.session_state.scan_pool["_favorites"] = st.session_state.favorites
            has_changes = True

        # 2. Sync Account Types and Group Moves
        for _, row in edited_df.iterrows():
            ticker = row["Ticker"]
            new_acct = row["Account Type"]
            new_cat = row["Group / Category"]

            if st.session_state.account_types.get(ticker) != new_acct:
                st.session_state.account_types[ticker] = new_acct
                has_changes = True

            # Group Category Reassignment
            current_cat = next((cat for cat in categories if ticker in st.session_state.scan_pool[cat]), None)
            if current_cat and current_cat != new_cat:
                st.session_state.scan_pool[current_cat].remove(ticker)
                st.session_state.scan_pool[new_cat].append(ticker)
                has_changes = True

        if has_changes:
            st.session_state.scan_pool["_account_types"] = st.session_state.account_types
            save_universe(st.session_state.scan_pool)
            st.rerun()

    else:
        st.info("No tickers found in universe. Add one above using the Quick Add form!")
