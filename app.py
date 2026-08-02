"""
app.py
======
Main Entry Point for the Dynamic ETF Portfolio Management Engine.
Imports configuration rules, Tier 1 screening logic, and Tier 2 technical scoring.
"""

import streamlit as st
import pandas as pd

# Imports from modular architecture
from config.portfolio import (
    DEFAULT_FAVORITES,
    DYNAMIC_SCAN_POOL,
    ALL_SCAN_TICKERS,
    DEFAULT_RISK_RULES,
    ACCOUNT_LOCATION_RULES,
)
from logic.tier1_screener import fetch_etf_fundamentals, run_tier1_screen
from logic.tier2_signals import run_tier2_scoring

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic ETF Strategy Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INITIALIZE SESSION STATE ---
if "favorites" not in st.session_state:
    st.session_state.favorites = set(DEFAULT_FAVORITES)

if "risk_rules" not in st.session_state:
    st.session_state.risk_rules = DEFAULT_RISK_RULES.copy()

# --- SIDEBAR: RISK PROFILE & ENGINE CONTROLS ---
st.sidebar.title("⚙️ Strategy Engine")

active_profile = st.sidebar.selectbox(
    "Active Risk Profile:",
    list(st.session_state.risk_rules.keys()),
    index=1,  # Default to Moderate
    help="Select the risk framework applied to Tier 1 fundamental screening.",
)

active_rules = st.session_state.risk_rules[active_profile]

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Active Rules Summary")
st.sidebar.caption(active_rules["description"])
st.sidebar.markdown(f"• **Max Expense Ratio:** `{active_rules['max_expense']}%`")
st.sidebar.markdown(f"• **Min Dividend Yield:** `{active_rules['min_yield']}%`")
st.sidebar.markdown(f"• **Max Beta vs S&P 500:** `{active_rules['max_beta']}`")
st.sidebar.markdown(f"• **Min AUM ($M):** `${active_rules['min_aum_m']}M`")
st.sidebar.markdown(f"• **Max 3Yr Volatility:** `{active_rules['max_volatility_3yr']}%`")

st.sidebar.markdown("---")
st.sidebar.metric("Saved Favorites Count", len(st.session_state.favorites))


# --- HEADER ---
st.title("⚡ Dynamic ETF Strategy Engine")
st.caption(
    "Modular ETF Screener: Screen candidates in Tier 1, route to optimal tax accounts, "
    "and generate tactical Buy/Sell/Hold signals in Tier 2."
)

# --- MAIN TABS ---
tab_rec, tab_favs, tab_tier2, tab_config = st.tabs([
    "🚀 Tier 1: Dynamic Recommendations",
    "⭐ My Favorites Watchlist",
    "📊 Tier 2: Tactical Buy/Sell Signals",
    "🛠️ Strategy Rule Configurator",
])

# ==============================================================================
# TAB 1: DYNAMIC RECOMMENDATIONS (TIER 1 SCREENER)
# ==============================================================================
with tab_rec:
    st.header(f"Live Tier 1 Market Suggestions ({active_profile} Profile)")
    st.caption(
        "Candidate ETFs screened against active risk limits and auto-routed "
        "to optimal tax locations."
    )

    col_btn1, col_btn2 = st.columns([2, 8])
    with col_btn1:
        if st.button("🔄 Refresh Market Scan", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Category Pool Filter
    category_filter = st.selectbox(
        "Scan Sub-Pool:",
        ["All Assets"] + list(DYNAMIC_SCAN_POOL.keys()),
        index=0,
    )

    if category_filter == "All Assets":
        scan_list = ALL_SCAN_TICKERS
    else:
        scan_list = DYNAMIC_SCAN_POOL[category_filter]

    with st.spinner("Fetching live fundamentals and evaluating Tier 1 rules..."):
        df_raw = fetch_etf_fundamentals(scan_list)
        df_screened = run_tier1_screen(df_raw, risk_profile_name=active_profile)

    if not df_screened.empty:
        # Group outputs into 3 tax account buckets
        t_taxable, t_roth, t_trad = st.tabs([
            "🏦 Taxable Brokerage",
            "📈 Roth IRA (Tax-Free Growth)",
            "🛡️ Traditional / Rollover IRA",
        ])

        buckets = [
            ("Taxable Brokerage", t_taxable),
            ("Roth IRA", t_roth),
            ("Traditional / Rollover IRA", t_trad),
        ]

        for bucket_name, tab_obj in buckets:
            with tab_obj:
                df_bucket = df_screened[df_screened["Target Account"] == bucket_name]
                if not df_bucket.empty:
                    for idx, row in df_bucket.iterrows():
                        ticker = row["Ticker"]
                        is_fav = ticker in st.session_state.favorites

                        c1, c2, c3, c4 = st.columns([1, 3, 3, 3])
                        with c1:
                            if is_fav:
                                st.success("⭐ Saved")
                            else:
                                if st.button(f"⭐ Fav", key=f"rec_fav_{ticker}"):
                                    st.session_state.favorites.add(ticker)
                                    st.rerun()
                        with c2:
                            st.markdown(f"**{ticker}** — *{row['Name']}*")
                            st.caption(f"Category: {row['Category']}")
                        with c3:
                            st.write(
                                f"Price: **${row['Price ($)']}** | Yield: **{row['Yield (%)']}%**"
                            )
                            st.write(f"Expense: **{row['Expense (%)']}%**")
                        with c4:
                            st.write(
                                f"Beta: **{row['Beta']}** | Vol 3Yr: **{row['Vol 3Yr (%)']}%**"
                            )
                            st.write(f"AUM: **${row['AUM ($M)']}M**")
                        st.markdown("---")
                else:
                    st.info(f"No candidates passed for {bucket_name} in this scan pool.")
    else:
        st.warning(
            "No ETFs passed the current risk profile rules. Try selecting a broader "
            "risk profile or editing limits in the Configurator tab."
        )


# ==============================================================================
# TAB 2: MY FAVORITES WATCHLIST
# ==============================================================================
with tab_favs:
    st.header("⭐ My Favorited Watchlist")
    st.caption("Manage saved tickers. Add new candidates or click **❌ UnFav** to prune.")

    col_add1, col_add2 = st.columns([3, 1])
    with col_add1:
        manual_ticker = (
            st.text_input("Quick Add Ticker Symbol:", placeholder="e.g. VTI")
            .upper()
            .strip()
        )
    with col_add2:
        st.write(" ")
        st.write(" ")
        if st.button("Add to Watchlist", use_container_width=True) and manual_ticker:
            st.session_state.favorites.add(manual_ticker)
            st.rerun()

    st.markdown("---")

    if st.session_state.favorites:
        fav_list = sorted(list(st.session_state.favorites))
        df_fav_raw = fetch_etf_fundamentals(fav_list)

        if not df_fav_raw.empty:
            df_fav_screened = run_tier1_screen(df_fav_raw, risk_profile_name=active_profile)
            if df_fav_screened.empty:
                df_fav_screened = df_fav_raw.copy()
                df_fav_screened["Target Account"] = "Uncategorized"

            for idx, row in df_fav_screened.iterrows():
                ticker = row["Ticker"]
                c_unfav, c_info, c_stats = st.columns([1, 4, 3])

                with c_unfav:
                    if st.button(f"❌ UnFav", key=f"unfav_{ticker}"):
                        st.session_state.favorites.remove(ticker)
                        st.toast(f"Removed {ticker} from favorites.")
                        st.rerun()

                with c_info:
                    st.markdown(f"### **{ticker}** — {row['Name']}")
                    st.caption(f"Account Bucket: **{row.get('Target Account', 'N/A')}**")

                with c_stats:
                    st.write(
                        f"Price: **${row['Price ($)']}** | Yield: **{row['Yield (%)']}%**"
                    )
                    st.write(
                        f"Expense: **{row['Expense (%)']}%** | Beta: **{row['Beta']}**"
                    )

                st.markdown("---")
    else:
        st.info("Your favorites list is empty. Add tickers above or from Recommendations.")


# ==============================================================================
# TAB 3: TIER 2 TACTICAL BUY/SELL SIGNALS
# ==============================================================================
with tab_tier2:
    st.header("📊 Tier 2 Technical & Momentum Scoring Engine")
    st.caption("Evaluates 200 SMA, 50 SMA, RSI (14), and MACD crossover signals across your favorited tickers.")

    if st.session_state.favorites:
        fav_list = sorted(list(st.session_state.favorites))

        with st.spinner("Calculating technical indicators and generating composite scores..."):
            df_tier2 = run_tier2_scoring(fav_list)

        if not df_tier2.empty:
            # Render Rating Cards Summary
            col_sb, col_b, col_h, col_s = st.columns(4)
            col_sb.metric("Strong Buy", len(df_tier2[df_tier2["Rating"] == "Strong Buy"]))
            col_b.metric("Buy", len(df_tier2[df_tier2["Rating"] == "Buy"]))
            col_h.metric("Hold", len(df_tier2[df_tier2["Rating"] == "Hold"]))
            col_s.metric("Sell", len(df_tier2[df_tier2["Rating"] == "Sell"]))

            st.markdown("---")

            # Detailed Output Table
            for idx, row in df_tier2.iterrows():
                ticker = row["Ticker"]
                rating = row["Rating"]
                score = row["Score"]

                # Color-code rating badges
                if rating == "Strong Buy":
                    badge = f"🟢 **{rating}** ({score}/100)"
                elif rating == "Buy":
                    badge = f"🔵 **{rating}** ({score}/100)"
                elif rating == "Hold":
                    badge = f"🟡 **{rating}** ({score}/100)"
                else:
                    badge = f"🔴 **{rating}** ({score}/100)"

                c1, c2, c3 = st.columns([2, 3, 5])
                with c1:
                    st.markdown(f"### {ticker}")
                    st.markdown(badge)
                with c2:
                    st.write(f"Price: **${row['Price ($)']}**")
                    st.write(f"RSI (14): **{row['RSI (14)']}**")
                    st.write(f"200 SMA: **${row['200 SMA']}** | 50 SMA: **${row['50 SMA']}**")
                with c3:
                    st.markdown("**Tactical Signals:**")
                    signals = row["Key Signals"].split(" | ")
                    for s in signals:
                        st.caption(f"• {s}")

                st.markdown("---")
        else:
            st.warning("Could not compute technical scores. Ensure tickers have valid price history.")
    else:
        st.info("Add tickers to your Favorites list to run Tier 2 technical scoring.")


# ==============================================================================
# TAB 4: STRATEGY RULE CONFIGURATOR
# ==============================================================================
with tab_config:
    st.header("🛠️ Risk Profile & Threshold Configurator")
    st.caption("Customize the exact numerical parameters that define each profile.")

    selected_edit_profile = st.selectbox(
        "Select Profile to Modify:",
        list(st.session_state.risk_rules.keys()),
    )

    p_rules = st.session_state.risk_rules[selected_edit_profile]

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.subheader("Fundamental Limits")
        new_max_exp = st.slider(
            "Max Expense Ratio (%)",
            0.03, 1.00, float(p_rules["max_expense"]), 0.01,
            key=f"exp_{selected_edit_profile}",
        )
        new_min_yield = st.slider(
            "Min Dividend Yield (%)",
            0.0, 6.0, float(p_rules["min_yield"]), 0.1,
            key=f"yield_{selected_edit_profile}",
        )
        new_min_aum = st.number_input(
            "Min AUM ($ Millions)",
            10, 5000, int(p_rules["min_aum_m"]), 50,
            key=f"aum_{selected_edit_profile}",
        )

    with c_c2:
        st.subheader("Risk & Volatility Limits")
        new_max_beta = st.slider(
            "Max Beta (vs S&P 500)",
            0.20, 2.50, float(p_rules["max_beta"]), 0.05,
            key=f"beta_{selected_edit_profile}",
        )
        new_max_vol = st.slider(
            "Max 3-Year Volatility (%)",
            5.0, 50.0, float(p_rules["max_volatility_3yr"]), 0.5,
            key=f"vol_{selected_edit_profile}",
        )

    if st.button(f"💾 Save Changes for {selected_edit_profile}", use_container_width=True):
        st.session_state.risk_rules[selected_edit_profile].update({
            "max_expense": new_max_exp,
            "min_yield": new_min_yield,
            "min_aum_m": new_min_aum,
            "max_beta": new_max_beta,
            "max_volatility_3yr": new_max_vol,
        })
        st.success(f"Updated parameters for {selected_edit_profile}!")
        st.rerun()
