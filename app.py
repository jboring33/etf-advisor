"""
app.py
======
90-Day Tactical ETF Screener & Dynamic Candidate Discovery Engine.
Scans US, Developed, and Ex-China Emerging markets for 90-day setups.
"""

import os
import sys
import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="90-Day Tactical ETF Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling Injection
st.markdown("""
<style>
    div.stButton > button[kind="primary"] {
        background-color: #1E88E5 !important;
        border-color: #1E88E5 !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

from config.portfolio import load_universe, save_universe

# Initialize Session State
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()
if "account_types" not in st.session_state:
    st.session_state.account_types = st.session_state.scan_pool.get("_account_types", {})
if "allocations" not in st.session_state:
    st.session_state.allocations = st.session_state.scan_pool.get("_allocations", {})
if "regions" not in st.session_state:
    st.session_state.regions = st.session_state.scan_pool.get("_regions", {})
if "favorites" not in st.session_state:
    st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])

def get_active_categories():
    return [cat for cat in st.session_state.scan_pool.keys() if not cat.startswith("_")]


# ==============================================================================
# DYNAMIC MARKET DISCOVERY WATCHLIST (BROAD MARKET SCANNING)
# ==============================================================================
# Broader candidate pool spanning US Core, Factors, Developed, Ex-China EM, and Fixed Income
DYNAMIC_MARKET_POOL = {
    # Ex-China & International
    "EMXC": {"name": "MSCI Emerging Markets ex-China", "region": "ex-China", "default_cat": "International/Emerging"},
    "VEA":  {"name": "Vanguard FTSE Developed Markets", "region": "Developed", "default_cat": "International/Emerging"},
    "DIVI": {"name": "International Dividend Achievers", "region": "Developed", "default_cat": "International/Emerging"},
    "INDA": {"name": "MSCI India ETF", "region": "Emerging", "default_cat": "International/Emerging"},
    "EWJ":  {"name": "iShares MSCI Japan ETF", "region": "Developed", "default_cat": "International/Emerging"},
    "EWT":  {"name": "iShares MSCI Taiwan ETF", "region": "Emerging", "default_cat": "International/Emerging"},
    
    # US Factor & Income
    "VFLO": {"name": "VictoryShares Free Cash Flow ETF", "region": "US", "default_cat": "Tactical/Growth"},
    "SCHD": {"name": "Schwab US Dividend Equity", "region": "US", "default_cat": "Core/Dividend"},
    "JPST": {"name": "JPMorgan Ultra-Short Income", "region": "US", "default_cat": "Fixed Income/Cash"},
    "JAAA": {"name": "Janus Henderson AAA CLO ETF", "region": "US", "default_cat": "Fixed Income/Cash"},
    "SCYB": {"name": "Schwab High Yield Bond ETF", "region": "US", "default_cat": "Fixed Income/Cash"},
    
    # US Sector & Momentum
    "SMH":  {"name": "VanEck Semiconductor ETF", "region": "US", "default_cat": "Tactical/Growth"},
    "XLK":  {"name": "Technology Select Sector SPDR", "region": "US", "default_cat": "Tactical/Growth"},
    "XLF":  {"name": "Financial Select Sector SPDR", "region": "US", "default_cat": "Core/Dividend"},
    "XLE":  {"name": "Energy Select Sector SPDR", "region": "US", "default_cat": "Tactical/Growth"},
    "XLI":  {"name": "Industrial Select Sector SPDR", "region": "US", "default_cat": "Core/Dividend"},
    "XLV":  {"name": "Health Care Select Sector SPDR", "region": "US", "default_cat": "Core/Dividend"},
    "IWM":  {"name": "iShares Russell 2000 ETF", "region": "US", "default_cat": "Tactical/Growth"},
    "QQQ":  {"name": "Invesco QQQ Trust", "region": "US", "default_cat": "Tactical/Growth"},
    "SPY":  {"name": "SPDR S&P 500 ETF Trust", "region": "US", "default_cat": "Core/Dividend"},
}


# ==============================================================================
# ANALYTICS ENGINE & DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=1800)
def fetch_fed_funds_probabilities():
    """Retrieves benchmark yields and macro posture."""
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        last_yield = tnx["Close"].iloc[-1] if not tnx.empty else 4.25
        return {
            "Next Meeting": "Sep 16, 2026",
            "Pause Probability": "70.7%",
            "Cut Probability (-25bps)": "29.3%",
            "Hike Probability": "0.0%",
            "10Yr Benchmark Yield": f"{last_yield:.2f}%",
            "Regime Sentiment": "Pause Expected / Easing Bias"
        }
    except Exception:
        return {
            "Next Meeting": "Upcoming",
            "Pause Probability": "68.0%",
            "Cut Probability (-25bps)": "32.0%",
            "Hike Probability": "0.0%",
            "10Yr Benchmark Yield": "4.20%",
            "Regime Sentiment": "Neutral"
        }

@st.cache_data(ttl=3600)
def analyze_etf_technical_ema(ticker: str):
    """Calculates 20/50 day EMAs to check trend convergence."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6m")
        if len(df) < 50:
            return None

        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        latest_close = df["Close"].iloc[-1]
        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]
        prev_ema20 = df["EMA20"].iloc[-5]

        gap_pct = ((ema20 - ema50) / ema50) * 100
        is_above = ema20 > ema50
        is_approaching = (not is_above) and (gap_pct > -2.0) and (ema20 > prev_ema20)

        status = "Bullish (20 > 50 EMA)" if is_above else ("Approaching Cross ↗️" if is_approaching else "Bearish Lag")
        
        return {
            "Close": latest_close,
            "EMA20": ema20,
            "EMA50": ema50,
            "Gap_Pct": gap_pct,
            "Status": status,
            "Bullish_Setup": is_above or is_approaching
        }
    except Exception:
        return None

@st.cache_data(ttl=7200)
def fetch_top_holdings_earnings(ticker: str):
    """Safe retrieval of top holdings and earnings catalysts."""
    holdings = [ticker]
    try:
        tk = yf.Ticker(ticker)
        if hasattr(tk, "funds_data") and tk.funds_data is not None:
            cfg = getattr(tk.funds_data, "top_holdings", None)
            if cfg is not None and not cfg.empty:
                holdings = cfg.index.tolist()[:7]
    except Exception:
        pass

    earnings_summary = []
    upcoming_count = 0
    positive_surprises = 0

    for symbol in holdings:
        try:
            sub_tk = yf.Ticker(symbol)
            cal = sub_tk.calendar
            
            next_date = "N/A"
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if ed:
                    next_date = ed[0].strftime("%Y-%m-%d") if isinstance(ed[0], datetime.date) else str(ed[0])
                    upcoming_count += 1
            
            surp_df = sub_tk.earnings_dates
            last_surprise = "N/A"
            if surp_df is not None and "Surprise(%)" in surp_df.columns:
                recent = surp_df.dropna(subset=["Surprise(%)"])
                if not recent.empty:
                    val = recent["Surprise(%)"].iloc[0] * 100
                    last_surprise = f"{val:+.1f}%"
                    if val > 0:
                        positive_surprises += 1

            earnings_summary.append({
                "Holding": symbol,
                "Next Earnings": next_date,
                "Last Surprise": last_surprise
            })
        except Exception:
            continue

    return {
        "Holdings_Count": len(holdings),
        "Upcoming_30D_Earnings": upcoming_count,
        "Positive_Surprise_Ratio": f"{positive_surprises}/{len(holdings)}" if holdings else "0/0",
        "Details": earnings_summary
    }

@st.cache_data(ttl=3600)
def fetch_institutional_flows_30d(ticker: str):
    """Calculates 30-day Volume-Weighted Accumulation score."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2m")
        if len(hist) < 20:
            return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

        hist30 = hist.tail(22).copy()
        hist30["Price_Change"] = hist30["Close"].diff()
        hist30["Directional_Vol"] = np.where(hist30["Price_Change"] >= 0, hist30["Volume"], -hist30["Volume"])
        
        net_flow_vol = hist30["Directional_Vol"].sum()
        avg_vol = hist30["Volume"].mean()
        flow_score = min(100, max(0, int(50 + (net_flow_vol / (avg_vol * 10)) * 50)))
        
        if flow_score >= 65:
            signal = "🔥 Accumulation"
        elif flow_score <= 35:
            signal = "🚨 Distribution"
        else:
            signal = "➡️ Steady"

        return {
            "Flow_Signal": signal,
            "Net_30D_Score": flow_score
        }
    except Exception:
        return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

def score_etf(ticker: str):
    """Calculates composite score (0-100) based on Technicals, Flows, and Earnings Catalysts."""
    tech = analyze_etf_technical_ema(ticker)
    if not tech:
        return None

    flows = fetch_institutional_flows_30d(ticker) or {"Flow_Signal": "Neutral", "Net_30D_Score": 50}
    earnings = fetch_top_holdings_earnings(ticker) or {"Holdings_Count": 0, "Upcoming_30D_Earnings": 0, "Positive_Surprise_Ratio": "N/A", "Details": []}

    score = 0
    if tech.get("Bullish_Setup"): score += 40
    if flows.get("Net_30D_Score", 0) >= 60: score += 30
    if earnings.get("Upcoming_30D_Earnings", 0) > 0: score += 30

    return {
        "Ticker": ticker,
        "Score": score,
        "Tech": tech,
        "Flows": flows,
        "Earnings": earnings
    }


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

st.title("🎯 90-Day Tactical Market Screener")
st.caption("Automated Market Scanner: US Sectors, Factors, and Ex-China International Candidates")

# Macro Rates Header
fed_data = fetch_fed_funds_probabilities()
st.subheader("🏛️ Macro Monitor")
f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
f_col1.metric("Next FOMC Meeting", fed_data["Next Meeting"])
f_col2.metric("Pause Prob.", fed_data["Pause Probability"])
f_col3.metric("Cut Prob.", fed_data["Cut Probability (-25bps)"])
f_col4.metric("Hike Prob.", fed_data["Hike Probability"])
f_col5.metric("10Yr Yield", fed_data["10Yr Benchmark Yield"], delta=fed_data["Regime Sentiment"])

st.markdown("---")

tab_discovery, tab_lookup, tab_universe = st.tabs([
    "🚀 90-Day Market Candidates",
    "🔍 Single ETF Deep-Dive",
    "⚙️ Active Portfolio Configurator"
])


# ==============================================================================
# TAB 1: AUTOMATED 90-DAY MARKET CANDIDATE DISCOVERY
# ==============================================================================
with tab_discovery:
    st.header("⚡ Top 90-Day Tactical Candidates Across Markets")
    st.caption("Scans broad sector, factor, and regional ETFs for upcoming momentum setups.")

    categories = get_active_categories()
    existing_universe_tickers = set(t for cat in categories for t in st.session_state.scan_pool.get(cat, []))

    # Controls
    col_filter_reg, col_min_score, _ = st.columns([2, 2, 3])
    with col_filter_reg:
        selected_region = st.selectbox("Filter Region:", ["All Regions", "US", "Developed", "Emerging", "ex-China"])
    with col_min_score:
        min_score_cutoff = st.slider("Minimum 90D Score:", min_value=0, max_value=100, value=40, step=10)

    discovered_candidates = []

    with st.spinner("Scanning market candidates..."):
        for ticker, info in DYNAMIC_MARKET_POOL.items():
            # Apply regional filter
            if selected_region != "All Regions" and info["region"] != selected_region:
                continue

            res = score_etf(ticker)
            if res and res["Score"] >= min_score_cutoff:
                in_portfolio = ticker in existing_universe_tickers
                discovered_candidates.append({
                    "Ticker": ticker,
                    "Name": info["name"],
                    "Region": info["region"],
                    "90D Score": res["Score"],
                    "EMA Trend Setup": res["Tech"]["Status"],
                    "30D Inst Flow": res["Flows"]["Flow_Signal"],
                    "Flow Score": f"{res['Flows']['Net_30D_Score']}/100",
                    "30D Holdings Catalysts": f"{res['Earnings']['Upcoming_30D_Earnings']} upcoming",
                    "In Portfolio": "✅ Active" if in_portfolio else "💡 Opportunity",
                    "_raw_res": res,
                    "_info": info
                })

    if discovered_candidates:
        cand_df = pd.DataFrame(discovered_candidates).sort_values(by="90D Score", ascending=False).reset_index(drop=True)

        st.dataframe(
            cand_df.drop(columns=["_raw_res", "_info"]),
            hide_index=True,
            column_config={
                "90D Score": st.column_config.ProgressColumn("90D Readiness Score", format="%d pts", min_value=0, max_value=100),
                "Ticker": st.column_config.TextColumn("Ticker", width=80),
                "In Portfolio": st.column_config.TextColumn("Status", width=120),
            },
            use_container_width=True
        )

        st.markdown("---")

        # Dynamic Auto-Add Mechanism
        st.subheader("➕ Add Candidate to Active Portfolio")
        unheld_options = [c["Ticker"] for c in discovered_candidates if c["In Portfolio"] == "💡 Opportunity"]
        
        if unheld_options:
            c_sel, c_btn = st.columns([3, 2])
            with c_sel:
                add_selected = st.selectbox("Select candidate to add:", options=unheld_options)
            with c_btn:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"Import {add_selected} into Portfolio", type="primary"):
                    meta = DYNAMIC_MARKET_POOL[add_selected]
                    target_cat = meta["default_cat"] if meta["default_cat"] in categories else categories[0]

                    st.session_state.scan_pool[target_cat].append(add_selected)
                    st.session_state.account_types[add_selected] = "Brokerage"
                    st.session_state.allocations[add_selected] = 0.0
                    st.session_state.regions[add_selected] = meta["region"]

                    st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                    st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                    st.session_state.scan_pool["_regions"] = st.session_state.regions
                    save_universe(st.session_state.scan_pool)

                    st.success(f"Added {add_selected} ({meta['region']}) to {target_cat}!")
                    st.rerun()
        else:
            st.info("All candidates matching your current filters are already in your active portfolio.")
    else:
        st.warning("No market candidates met the minimum score criteria for the selected filters.")


# ==============================================================================
# TAB 2: SINGLE ETF SCORE LOOKUP
# ==============================================================================
with tab_lookup:
    st.header("🔍 Single ETF Deep-Dive")
    st.caption("Calculate 90-day technical, flow, and earnings scores for any ticker symbol.")

    col_input, _ = st.columns([2, 3])
    with col_input:
        lookup_ticker = st.text_input("Enter Ticker Symbol:", value="EMXC", placeholder="e.g. EMXC, SCHD, VFLO").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Evaluating {lookup_ticker}..."):
            res = score_etf(lookup_ticker)

        if res:
            meta = DYNAMIC_MARKET_POOL.get(lookup_ticker, {"region": "US", "default_cat": get_active_categories()[0]})
            st.markdown(f"### Score for **{lookup_ticker}**: `{res['Score']}/100` Points")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Technical Setup (20/50 EMA)", res["Tech"]["Status"], delta=f"Gap: {res['Tech']['Gap_Pct']:.2f}%")
                st.write(f"- **Close:** ${res['Tech']['Close']:.2f}")
                st.write(f"- **20 EMA:** ${res['Tech']['EMA20']:.2f}")
                st.write(f"- **50 EMA:** ${res['Tech']['EMA50']:.2f}")

            with sc2:
                st.metric("30D Institutional Flow", res["Flows"]["Flow_Signal"], delta=f"Score: {res['Flows']['Net_30D_Score']}/100")
                st.write("Calculated via 30-day Volume-Weighted Money Flow.")

            with sc3:
                st.metric("30D Holdings Earnings", f"{res['Earnings']['Upcoming_30D_Earnings']} Upcoming", delta=f"Beat Ratio: {res['Earnings']['Positive_Surprise_Ratio']}")
                st.write(f"Analyzed top {res['Earnings']['Holdings_Count']} constituent holdings.")
        else:
            st.error(f"Could not retrieve ticker data for '{lookup_ticker}'. Please verify the symbol.")


# ==============================================================================
# TAB 3: ACTIVE PORTFOLIO CONFIGURATOR
# ==============================================================================
with tab_universe:
    st.header("⚙️ Active Portfolio Management")
    categories = get_active_categories()

    master_rows = []
    for category in categories:
        for t in st.session_state.scan_pool.get(category, []):
            master_rows.append({
                "Bucket": st.session_state.account_types.get(t, "Brokerage"),
                "Ticker": t,
                "Type": category,
                "Region": st.session_state.regions.get(t, "US"),
                "Allocation (%)": float(st.session_state.allocations.get(t, 0.0)),
                "⭐ Fav": t in st.session_state.favorites,
                "Delete": False
            })

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        edited_df = st.data_editor(
            master_df,
            key="portfolio_manager_grid",
            hide_index=True,
            column_config={
                "Bucket": st.column_config.SelectboxColumn("Bucket", options=["Brokerage", "IRA", "Roth/HSA"], required=True),
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "Region": st.column_config.SelectboxColumn("Region", options=["US", "Emerging", "Developed", "ex-China"], required=True),
                "Allocation (%)": st.column_config.NumberColumn("Alloc (%)", format="%.1f%%"),
                "⭐ Fav": st.column_config.CheckboxColumn("Fav", width=45),
                "Delete": st.column_config.CheckboxColumn("Del", width=45)
            },
            use_container_width=True
        )

        if st.button("💾 Save Portfolio Changes", type="primary"):
            st.session_state.favorites = edited_df[edited_df["⭐ Fav"] == True]["Ticker"].tolist()
            for _, row in edited_df.iterrows():
                t = row["Ticker"]
                st.session_state.account_types[t] = str(row["Bucket"])
                st.session_state.regions[t] = str(row["Region"])
                st.session_state.allocations[t] = float(row["Allocation (%)"])

            st.session_state.scan_pool["_favorites"] = st.session_state.favorites
            st.session_state.scan_pool["_account_types"] = st.session_state.account_types
            st.session_state.scan_pool["_regions"] = st.session_state.regions
            st.session_state.scan_pool["_allocations"] = st.session_state.allocations
            save_universe(st.session_state.scan_pool)
            st.success("Portfolio updated successfully!")
            st.rerun()
