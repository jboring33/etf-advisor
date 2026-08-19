# app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Force root directory into sys.path to resolve module discovery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Safe imports for internal modules
try:
    import logic.tier1_screener as tier1
except ImportError:
    import tier1_screener as tier1

try:
    import logic.tier2_signals as tier2
except ImportError:
    import tier2_signals as tier2

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ETF Tactical Screener & Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ETF Tactical Screener & Advisor")
st.markdown("""
*A quantitative ETF evaluation dashboard designed to screen candidates across asset classes,
map tax efficiency by account type, and analyze composite technical and fundamental scores.*
""")

# Default Watchlist Tickers
DEFAULT_TICKERS = ["VFLO", "SCHD", "SCYB", "JPST", "JAAA", "VEA", "DIVI", "EMXC"]

# Initialize Session State
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_TICKERS

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & CUSTOM TABLE LAYOUT
# -----------------------------------------------------------------------------
def render_fund_table(df_subset, bucket_name):
    """
    Renders fund metrics grouped by tax account bucket.
    Includes hover flyover for fund names and places Favorite / Delete controls 
    at the end of each row.
    """
    if df_subset.empty:
        st.info(f"No ETF candidates currently assigned to **{bucket_name}**.")
        return

    st.caption(f"Showing **{len(df_subset)}** ETF candidate(s) categorized under **{bucket_name}**.")

    for idx, row in df_subset.iterrows():
        ticker = row.get("Ticker", idx)
        name = row.get("Name", "N/A")
        category = row.get("Category", "N/A")
        region = row.get("Region", "N/A")
        exp_ratio = row.get("Expense Ratio", "N/A")
        div_yield = row.get("Dividend Yield", "N/A")
        beta = row.get("Beta", "N/A")
        aum = row.get("AUM ($M)", "N/A")

        with st.container():
            # Layout: Details first, trailing Favorite & Delete controls at the end
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1.2, 2.0, 1.5, 1.2, 1.2, 1.0, 0.8])

            with col1:
                # Ticker with Name Flyover/Tooltip
                st.markdown(f"**{ticker}**", help=f"Full Name: {name}")
                st.caption(f"{category}")

            with col2:
                st.write(f"**Region:** {region}")

            with col3:
                st.write(f"**AUM:** {aum}")

            with col4:
                st.write(f"**Exp Ratio:** {exp_ratio}")

            with col5:
                st.write(f"**Yield:** {div_yield}")

            with col6:
                # Favorite Toggle
                is_fav = ticker in st.session_state.favorites
                fav_label = "★ Fav" if is_fav else "☆ Fav"
                if st.button(fav_label, key=f"fav_{bucket_name}_{ticker}"):
                    if is_fav:
                        st.session_state.favorites.remove(ticker)
                    else:
                        st.session_state.favorites.append(ticker)
                    st.rerun()

            with col7:
                # Delete Toggle at end of row
                if st.button("🗑️", key=f"del_{bucket_name}_{ticker}", help=f"Remove {ticker} from watchlist"):
                    if ticker in st.session_state.watchlist:
                        st.session_state.watchlist.remove(ticker)
                        st.rerun()

            st.divider()

# -----------------------------------------------------------------------------
# SIDEBAR - CONFIGURATION & WATCHLIST MANAGEMENT
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Watchlist Management")
user_input = st.sidebar.text_area(
    "Enter Ticker Symbols (separated by space or comma):",
    value=" ".join(st.session_state.watchlist),
    height=100
)

# Cleaned, valid string parsing to avoid literal syntax errors
tickers_list = [
    t.strip().upper() 
    for t in user_input.replace(",", " ").split() 
    if t.strip()
]

if st.sidebar.button("Update Watchlist"):
    st.session_state.watchlist = tickers_list
    st.sidebar.success("Watchlist updated successfully!")

if st.session_state.favorites:
    st.sidebar.subheader("⭐ Favorites")
    st.sidebar.write(", ".join(st.session_state.favorites))

# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Batch Universe Screener", 
    "🔍 Single Symbol Scorecard", 
    "⚙️ Points Configurator"
])

# -----------------------------------------------------------------------------
# TAB 1: BATCH UNIVERSE SCREENER (ACCOUNT BUCKET TABS)
# -----------------------------------------------------------------------------
with tab1:
    st.header("Batch Universe Screener")
    st.markdown("""
    Screen candidate ETFs across fundamental quality thresholds and view results grouped
    by tax account efficiency.
    """)

    if st.button("🚀 Run Batch Screener", type="primary"):
        with st.spinner("Fetching market data, fundamentals, and running indicators..."):
            try:
                if hasattr(tier1, "run_tier1_screen"):
                    results_df = tier1.run_tier1_screen(st.session_state.watchlist)
                elif hasattr(tier1, "run_batch_screener"):
                    results_df = tier1.run_batch_screener(st.session_state.watchlist)
                else:
                    results_df = pd.DataFrame()

                st.session_state["screener_results"] = results_df
            except Exception as e:
                st.error(f"Error running batch screener: {str(e)}")

    # Account Bucket Tab Presentation
    if "screener_results" in st.session_state and isinstance(st.session_state["screener_results"], pd.DataFrame):
        df_results = st.session_state["screener_results"]

        if not df_results.empty:
            bucket_taxable, bucket_roth, bucket_trad = st.tabs([
                "💼 Taxable Brokerage", 
                "🛡️ Roth IRA", 
                "🏛️ Traditional IRA"
            ])

            with bucket_taxable:
                st.subheader("Taxable Brokerage Candidates")
                st.info("Focuses on high tax efficiency, broad core equities, and low-turnover funds.")
                sub_df = df_results[df_results["Bucket"].str.contains("Taxable", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Taxable Brokerage")

            with bucket_roth:
                st.subheader("Roth IRA Candidates")
                st.info("Focuses on high-growth assets, emerging markets, and total return drivers for tax-free compounding.")
                sub_df = df_results[df_results["Bucket"].str.contains("Roth", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Roth IRA")

            with bucket_trad:
                st.subheader("Traditional IRA Candidates")
                st.info("Focuses on high yield, dividend focus, and income drivers sheltered from annual tax drag.")
                sub_df = df_results[df_results["Bucket"].str.contains("Traditional", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Traditional IRA")
        else:
            st.warning("No screening results returned. Click 'Run Batch Screener' to load data.")

# -----------------------------------------------------------------------------
# TAB 2: SINGLE SYMBOL SCORECARD & COMMENTARY
# -----------------------------------------------------------------------------
with tab2:
    st.header("Single Symbol Scorecard")
    st.markdown("Detailed rule-by-rule breakdown, technical indicators, and tactical commentary for individual ETFs.")
    
    selected_ticker = st.selectbox(
        "Select Ticker for Detailed Breakdown:",
        options=st.session_state.watchlist
    )

    if selected_ticker:
        with st.spinner(f"Generating detailed scorecard for {selected_ticker}..."):
            try:
                if hasattr(tier2, "generate_symbol_scorecard"):
                    scorecard = tier2.generate_symbol_scorecard(selected_ticker)
                else:
                    scorecard = {}
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Ticker", selected_ticker)
                    st.metric("Composite Score", f"{scorecard.get('total_score', 0)} / 100")
                    st.markdown(f"**Target Allocation:** {scorecard.get('allocation_recommendation', 'N/A')}")
                
                with col2:
                    st.subheader("Rule Breakdown")
                    if "breakdown" in scorecard:
                        st.table(pd.DataFrame(scorecard["breakdown"]))
                    else:
                        st.info("Rule breakdown available once signals run.")

                if "commentary" in scorecard:
                    st.subheader("Tactical Commentary")
                    st.write(scorecard["commentary"])

            except Exception as e:
                st.error(f"Error generating scorecard for {selected_ticker}: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 3: POINTS CONFIGURATOR
# -----------------------------------------------------------------------------
with tab3:
    st.header("Points & Signal Weight Configurator")
    st.markdown("Adjust weight allocations across quantitative screening criteria.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Technical Signal Weights")
        st.slider("EMA Trend Weight", 0, 30, 20, key="weight_ema")
        st.slider("Relative Strength (RSI/ROC)", 0, 30, 20, key="weight_rs")
        st.slider("Volume Expansion", 0, 20, 10, key="weight_vol")

    with col_b:
        st.subheader("Fundamental & Macro Weights")
        st.slider("Yield / Fundamental Quality", 0, 30, 25, key="weight_fund")
        st.slider("Macro Regime Overlay", 0, 25, 25, key="weight_macro")

    st.success("Configuration saved automatically to session state.")
