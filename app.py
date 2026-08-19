import streamlit as st
import pandas as pd
import numpy as np

from logic.tier1_screener import run_batch_screener, fetch_symbol_data
from logic.tier2_signals import generate_symbol_scorecard

# Page Configuration
st.set_page_config(
    page_title="ETF Tactical Screener & Advisor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ETF Tactical Screener & Advisor")

# Default Watchlist Tickers
DEFAULT_TICKERS = ["VFLO", "SCHD", "SCYB", "JPST", "JAAA", "VEA", "DIVI", "EMXC"]

# Initialize Session State Watchlist
if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_TICKERS

# Sidebar - Configuration & Watchlist Management
st.sidebar.header("⚙️ Configuration")

# Watchlist Input
st.sidebar.subheader("Watchlist Management")
user_input = st.sidebar.text_area(
    "Enter Ticker Symbols (separated by space or comma):",
    value=" ".join(st.session_state.watchlist),
    height=100
)

# FIXED: Sanitized string parsing to prevent unterminated string literal error
tickers_list = [
    t.strip().upper() 
    for t in user_input.replace(",", " ").split() 
    if t.strip()
]

if st.sidebar.button("Update Watchlist"):
    st.session_state.watchlist = tickers_list
    st.sidebar.success("Watchlist updated successfully!")

# Tab Layout (Historical Tracker Tab Removed)
tab1, tab2, tab3 = st.tabs([
    "📊 Batch Universe Screener", 
    "🔍 Single Symbol Scorecard", 
    "⚙️ Points Configurator"
])

# -----------------------------------------------------------------------------
# TAB 1: BATCH UNIVERSE SCREENER
# -----------------------------------------------------------------------------
with tab1:
    st.header("Batch Universe Screener")
    st.write("Screen your active watchlist across key tactical rules and signals.")

    if st.button("🚀 Run Batch Screener", type="primary"):
        with st.spinner("Fetching market data and running indicators..."):
            try:
                results_df = run_batch_screener(st.session_state.watchlist)
                
                if not results_df.empty:
                    st.dataframe(
                        results_df.style.background_gradient(
                            cmap="Greens", subset=["Total Score"]
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("No data retrieved for the selected tickers.")
            except Exception as e:
                st.error(f"Error running batch screener: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 2: SINGLE SYMBOL SCORECARD
# -----------------------------------------------------------------------------
with tab2:
    st.header("Single Symbol Scorecard")
    
    selected_ticker = st.selectbox(
        "Select Ticker for Detailed Breakdown:",
        options=st.session_state.watchlist
    )

    if selected_ticker:
        with st.spinner(f"Generating detailed scorecard for {selected_ticker}..."):
            try:
                scorecard = generate_symbol_scorecard(selected_ticker)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Ticker", selected_ticker)
                    st.metric("Composite Score", f"{scorecard.get('total_score', 0)} / 100")
                
                with col2:
                    st.subheader("Rule Breakdown")
                    if "breakdown" in scorecard:
                        st.table(pd.DataFrame(scorecard["breakdown"]))
                    else:
                        st.info("Detailed rule breakdown not available.")
            except Exception as e:
                st.error(f"Error generating scorecard for {selected_ticker}: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 3: POINTS CONFIGURATOR
# -----------------------------------------------------------------------------
with tab3:
    st.header("Points & Signal Weight Configurator")
    st.write("Adjust weight allocations across screening criteria.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.slider("EMA Trend Weight", 0, 30, 20, key="weight_ema")
        st.slider("Relative Strength (RSI/ROC)", 0, 30, 20, key="weight_rs")
        st.slider("Volume Expansion", 0, 20, 10, key="weight_vol")

    with col_b:
        st.slider("Yield / Fundamental Quality", 0, 30, 25, key="weight_fund")
        st.slider("Macro Regime Overlay", 0, 25, 25, key="weight_macro")

    st.success("Configuration saved automatically to session state.")
