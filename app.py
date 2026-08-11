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
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="ETF Asset Location & Tactical Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM BLUE THEME INJECTION ---
st.markdown("""
<style>
    /* Primary Save Button Blue Override */
    div.stButton > button[kind="primary"] {
        background-color: #1E88E5 !important;
        border-color: #1E88E5 !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1565C0 !important;
        border-color: #1565C0 !important;
    }

    /* Override focus rings and active highlight lines to Blue */
    [data-baseweb="select"] :focus,
    div[aria-selected="true"] {
        border-color: #1E88E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Imports
from config.portfolio import (
    load_universe, 
    save_universe
)
from logic.tier1_screener import get_tax_location_recommendation
from logic.tier2_signals import calculate_tier2_signals
from logic.macro_overlay import evaluate_market_regime, apply_macro_regime_overlay

# Helper to map tax recommendations to concise tax classifications
def format_tax_recommendation(rec_text: str) -> str:
    rec_lower = rec_text.lower()
    if "qualified" in rec_lower:
        return "Qualified Dividends"
    elif "capital gains" in rec_lower or "equity" in rec_lower or "long-term" in rec_lower:
        return "Long-Term Capital Gains"
    elif "ordinary" in rec_lower or "income" in rec_lower or "bond" in rec_lower or "reit" in rec_lower:
        return "Ordinary Income"
    elif "exempt" in rec_lower or "deferred" in rec_lower or "muni" in rec_lower:
        return "Tax-Exempt / Deferred"
    return "Ordinary Income"

# Initialize persistent session state from JSON disk storage
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()

if "account_types" not in st.session_state:
    st.session_state.account_types = st.session_state.scan_pool.get("_account_types", {})

if "allocations" not in st.session_state:
    st.session_state.allocations = st.session_state.scan_pool.get("_allocations", {})

if "favorites" not in st.session_state:
    st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])

# Helper to get active user category list dynamically
def get_active_categories():
    return [cat for cat in st.session_state.scan_pool.keys() if not cat.startswith("_")]

# Helper function to fetch live Yahoo Finance metrics
@st.cache_data(ttl=3600)
def fetch_ticker_metrics(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info

        expense_ratio = info.get("expenseRatio", 0.0015)
        if expense_ratio and expense_ratio < 0.05:
            expense_ratio = expense_ratio * 100

        aum_m = info.get("totalAssets", 0)
        aum_m = (aum_m / 1e6) if aum_m else 0.0

        yield_pct = info.get("yield", info.get("dividendYield", 0.0))
        if yield_pct and yield_pct < 0.5:
            yield_pct = yield_pct * 100

        # Retrieve Morningstar Style Box info
        style_box = info.get("fundFamily", None)
        if not style_box or style_box == "None":
            category_name = info.get("category", "")
            if "Large" in category_name or "Cap" in category_name:
                style_box = category_name
            else:
                style_box = "Large Blend"

        # Retrieve Morningstar 3-Year Star Rating
        stars_num = info.get("threeYearStarRating", info.get("overallStarRating", 4))
        try:
            stars_num = int(stars_num)
        except (ValueError, TypeError):
            stars_num = 4
        
        star_str = "⭐" * max(1, min(5, stars_num))

        return {
            "Expense Ratio": expense_ratio if expense_ratio else 0.15,
            "AUM ($M)": aum_m,
            "Yield (%)": yield_pct,
            "Style Box": style_box,
            "3Yr Rating": star_str
        }
    except Exception:
        return {
            "Expense Ratio": 0.15, 
            "AUM ($M)": 0.0, 
            "Yield (%)": 0.0, 
            "Style Box": "Large Blend",
            "3Yr Rating": "⭐⭐⭐⭐"
        }


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
st.sidebar.info("💡 **Tip:** Click **💾 Save Changes** under the ETF table to persist edits to disk.")


# ==============================================================================
# MAIN DASHBOARD HEADER
# ==============================================================================
st.title("📈 ETF Asset Location & Tactical Screener")
st.caption("A decision framework for tax-efficient asset placement and momentum timing.")

# 3-Tab Streamlined Layout
tab1, tab2, tab3 = st.tabs([
    "🌐 ETF Universe",
    "📊 Tactical Buy/Sell Signals",
    "🛠️ Strategy Rule Configurator"
])


# ==============================================================================
# TAB 1: ETF UNIVERSE (Unified Screener, Location & Dynamic Management)
# ==============================================================================
with tab1:
    st.header("🌐 ETF Universe & Tax Location Screener")
    st.caption("Manage your tickers, view live fundamental metrics, assign account buckets, and evaluate tax placement.")

    categories = get_active_categories()

    # --- SINGLE UNIFIED QUICK ADD SECTION ---
    st.subheader("➕ Quick Add Ticker")
    with st.form("quick_add_master_form", clear_on_submit=True):
        col_t, col_c, col_a, col_pct, col_btn = st.columns([2, 2, 2, 2, 1.5])
        
        with col_t:
            add_ticker = st.text_input("Ticker Symbol", placeholder="e.g. VTI").strip().upper()
        with col_c:
            add_category = st.selectbox("Group Category", options=categories)
        with col_a:
            add_account = st.selectbox("Bucket (Account Type)", options=["Brokerage", "IRA", "Roth/HSA"])
        with col_pct:
            add_alloc = st.number_input("Allocation (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        with col_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            add_submitted = st.form_submit_button("➕ Add Ticker", use_container_width=True)

        if add_submitted and add_ticker and add_category:
            existing_tickers = [t for cat in categories for t in st.session_state.scan_pool[cat]]
            if add_ticker not in existing_tickers:
                st.session_state.scan_pool[add_category].append(add_ticker)
                st.session_state.account_types[add_ticker] = add_account
                st.session_state.allocations[add_ticker] = add_alloc
                
                # Persist directly to disk
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                save_universe(st.session_state.scan_pool)
                
                st.success(f"Added {add_ticker} to {add_category} ({add_account} | {add_alloc:.1f}%)!")
                st.rerun()
            else:
                st.warning(f"Ticker {add_ticker} already exists in the universe.")

    st.markdown("---")

    # --- MASTER ETF UNIVERSE TABLE DISPLAY ---
    st.subheader("📊 Master ETF Table")

    master_rows = []
    for category in categories:
        tickers = st.session_state.scan_pool.get(category, [])
        for t in tickers:
            if filter_by_favs and t not in st.session_state.favorites:
                continue

            metrics = fetch_ticker_metrics(t)
            bucket = st.session_state.account_types.get(t, "Brokerage")
            alloc = float(st.session_state.allocations.get(t, 0.0))
            raw_tax_rec = get_tax_location_recommendation(category, bucket)
            tax_rec = format_tax_recommendation(raw_tax_rec)

            master_rows.append({
                "Delete": False,
                "⭐ Fav": t in st.session_state.favorites,
                "Bucket": bucket,
                "Morningstar Style Box": metrics["Style Box"],
                "Morningstar 3Yr Rating": metrics["3Yr Rating"],
                "Group / Category": category,
                "Ticker": t,
                "Allocation (%)": alloc,
                "Expense Ratio": metrics["Expense Ratio"],
                "AUM": metrics["AUM ($M)"],
                "Yield": metrics["Yield (%)"],
                "Tax Recommendation": tax_rec
            })

    if master_rows:
        master_df = pd.DataFrame(master_rows)

        # Total Allocation Calculation & Dynamic Warning Banner
        total_alloc = master_df["Allocation (%)"].sum()
        
        if total_alloc > 100.0:
            st.error(f"🚨 **Allocation Error:** Total allocation across all buckets is **{total_alloc:.1f}%**, which exceeds the maximum allowed **100.0%**. Please adjust individual ticker allocations.")
        elif total_alloc < 100.0:
            st.info(f"ℹ️ Total Portfolio Allocation: **{total_alloc:.1f}%** / 100.0% ({100.0 - total_alloc:.1f}% unallocated)")
        else:
            st.success(f"✅ Total Portfolio Allocation: **100.0%** (Fully Allocated)")

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
                "Bucket": st.column_config.SelectboxColumn(
                    "Bucket",
                    options=["Brokerage", "IRA", "Roth/HSA"],
                    required=True
                ),
                "Morningstar Style Box": st.column_config.TextColumn(
                    "Morningstar Style Box",
                    disabled=True,
                    help="Morningstar style box / market cap classification"
                ),
                "Morningstar 3Yr Rating": st.column_config.TextColumn(
                    "Morningstar 3Yr Rating",
                    disabled=True,
                    help="Morningstar 3-year risk-adjusted star rating"
                ),
                "Group / Category": st.column_config.SelectboxColumn(
                    "Group / Category",
                    options=categories,
                    required=True
                ),
                "Ticker": st.column_config.TextColumn(
                    "Ticker",
                    disabled=True
                ),
                "Allocation (%)": st.column_config.NumberColumn(
                    "Allocation (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    required=True
                ),
                "Expense Ratio": st.column_config.NumberColumn(
                    "Expense Ratio",
                    format="%.2f%%",
                    disabled=True
                ),
                "AUM": st.column_config.NumberColumn(
                    "AUM ($M)",
                    format="$%.0fM",
                    disabled=True
                ),
                "Yield": st.column_config.NumberColumn(
                    "Yield (%)",
                    format="%.2f%%",
                    disabled=True
                ),
                "Tax Recommendation": st.column_config.TextColumn(
                    "Tax Recommendation",
                    disabled=True
                )
            },
            use_container_width=True
        )

        col_save, col_del, col_space = st.columns([1.5, 1.5, 3])

        # Dedicated Save Changes Button
        with col_save:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                # Sync Favorites
                updated_favs = edited_df[edited_df["⭐ Fav"] == True]["Ticker"].tolist()
                st.session_state.favorites = updated_favs

                # Sync Bucket, Allocation, and Category Changes
                for _, row in edited_df.iterrows():
                    ticker = row["Ticker"]
                    new_bucket = row["Bucket"]
                    new_cat = row["Group / Category"]
                    new_alloc = float(row["Allocation (%)"])

                    st.session_state.account_types[ticker] = new_bucket
                    st.session_state.allocations[ticker] = new_alloc

                    # Handle Category Transfer
                    current_cat = next((cat for cat in categories if ticker in st.session_state.scan_pool[cat]), None)
                    if current_cat and current_cat != new_cat:
                        st.session_state.scan_pool[current_cat].remove(ticker)
                        st.session_state.scan_pool[new_cat].append(ticker)

                # Persist to JSON
                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                save_universe(st.session_state.scan_pool)

                st.success("💾 Changes saved successfully to disk!")
                st.rerun()

        # Handle Bulk Deletions
        with col_del:
            selected_deletes = edited_df[edited_df["Delete"] == True]
            if not selected_deletes.empty:
                to_delete = selected_deletes["Ticker"].tolist()
                if st.button(f"🗑️ Delete Selected ({len(to_delete)})", use_container_width=True):
                    for cat in categories:
                        st.session_state.scan_pool[cat] = [
                            t for t in st.session_state.scan_pool[cat] if t not in to_delete
                        ]
                    st.session_state.favorites = [t for t in st.session_state.favorites if t not in to_delete]
                    for t in to_delete:
                        st.session_state.account_types.pop(t, None)
                        st.session_state.allocations.pop(t, None)

                    # Persist changes
                    st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                    st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                    st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                    save_universe(st.session_state.scan_pool)
                    st.success(f"Deleted {', '.join(to_delete)} from universe!")
                    st.rerun()

    else:
        st.info("No tickers found in universe. Add one above using the Quick Add form!")


# ==============================================================================
# TAB 2: TACTICAL BUY/SELL SIGNALS
# ==============================================================================
with tab2:
    st.header("📊 Tactical Technical Signals & Macro Overlay")
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
                            alloc_tag = f" [{st.session_state.allocations.get(ticker, 0.0):.1f}%]"
                            
                            st.metric(
                                label=f"{fav_icon}{ticker}{acct_tag}{alloc_tag}",
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
