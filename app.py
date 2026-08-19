# app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Force root directory into sys.path to resolve module discovery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports for internal modules with fallback protection
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
# HELPER FUNCTIONS & TABLE RENDERING
# -----------------------------------------------------------------------------
def render_fund_table(df_subset, bucket_name):
    """
    Renders structured fund metrics grouped by tax account bucket with interactive
    controls for favoriting and removing tickers directly from the view.
    """
    if df_subset.empty:
        st.info(f"No ETF candidates currently assigned to **{bucket_name}**.")
        return

    # Commentary / Overview Header for the Table
    st.caption(f"Showing **{len(df_subset)}** ETF candidate(s) optimized for **{bucket_name}**.")

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
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1.2, 2.5, 1.5, 1.2, 1.2, 1.0, 1.0])

            with col1:
                st.markdown(f"**{ticker}**", help=f"{ticker}: {name}")
                st.caption(category)

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
                # Delete Toggle
                if st.button("🗑️", key=f"del_{bucket_name}_{ticker}", help=f"Remove {ticker} from current view"):
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

# FIXED: Sanitized string parsing to prevent syntax errors
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
# TAB 1: BATCH UNIVERSE SCREENER (ACCOUNT BUCKETS & TABLES)
# -----------------------------------------------------------------------------
with tab1:
    st.header("Batch Universe Screener")
    st.markdown("""
    Screen your active watchlist across fundamental thresholds, risk parameters, and tax allocation buckets.
    Select an account type below to view candidate ETFs categorized specifically for that vehicle.
    """)

    if st.button("🚀 Run Batch Screener", type="primary"):
        with st.spinner("Fetching market data, fundamentals, and running indicators..."):
            try:
                # Fetch screened DataFrame from tier1 module
                if hasattr(tier1, "run_tier1_screen"):
                    results_df = tier1.run_tier1_screen(st.session_state.watchlist)
                elif hasattr(tier1, "run_batch_screener"):
                    results_df = tier1.run_batch_screener(st.session_state.watchlist)
                else:
                    results_df = pd.DataFrame()

                st.session_state["screener_results"] = results_df
            except Exception as e:
                st.error(f"Error running batch screener: {str(e)}")

    # Display Bucket Tabs if screener data is available
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
                st.info("Focuses on high tax efficiency, broad core equities, low turnover, and tax-managed funds.")
                sub_df = df_results[df_results["Bucket"].str.contains("Taxable", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Taxable Brokerage")

            with bucket_roth:
                st.subheader("Roth IRA Candidates")
                st.info("Focuses on high-growth assets, emerging markets, and high total-return drivers for tax-free compounding.")
                sub_df = df_results[df_results["Bucket"].str.contains("Roth", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Roth IRA")

            with bucket_trad:
                st.subheader("Traditional IRA Candidates")
                st.info("Focuses on high-yield, dividend, fixed-income, and ordinary income-generating funds sheltered from annual tax drag.")
                sub_df = df_results[df_results["Bucket"].str.contains("Traditional", case=False, na=False)] if "Bucket" in df_results.columns else df_results
                render_fund_table(sub_df, "Traditional IRA")
        else:
            st.warning("No screening results available. Click 'Run Batch Screener' to load data.")

# -----------------------------------------------------------------------------
# TAB 2: SINGLE SYMBOL SCORECARD
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
                    st.subheader("Rule Breakdown & Commentary")
                    if "breakdown" in scorecard:
                        st.table(pd.DataFrame(scorecard["breakdown"]))
                    else:
                        st.info("Detailed rule breakdown table available upon running signals.")

                if "commentary" in scorecard:
                    st.subheader("Tactical Investment Commentary")
                    st.write(scorecard["commentary"])

            except Exception as e:
                st.error(f"Error generating scorecard for {selected_ticker}: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 3: POINTS CONFIGURATOR
# -----------------------------------------------------------------------------
with tab3:
    st.header("Points & Signal Weight Configurator")
    st.markdown("Adjust weight allocations and scoring criteria for tactical screeners.")

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

    st.success("Configuration weights updated and saved to session state.")
