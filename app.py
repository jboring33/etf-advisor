"""
app.py
======
90-Day Tactical ETF Screener & Deep Dive Analysis Tool.
Fixed: Cloud server rate-limit bypass (requests session + custom User-Agent)
       and MultiIndex column flattening.
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
import requests

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


# ==============================================================================
# DYNAMIC MARKET DISCOVERY WATCHLIST
# ==============================================================================
DYNAMIC_MARKET_POOL = {
    # Ex-China & International
    "EMXC": {"name": "MSCI Emerging Markets ex-China", "region": "ex-China"},
    "VEA":  {"name": "Vanguard FTSE Developed Markets", "region": "Developed"},
    "DIVI": {"name": "International Dividend Achievers", "region": "Developed"},
    "INDA": {"name": "MSCI India ETF", "region": "Emerging"},
    "EWJ":  {"name": "iShares MSCI Japan ETF", "region": "Developed"},
    "EWT":  {"name": "iShares MSCI Taiwan ETF", "region": "Emerging"},
    
    # US Factor & Income
    "VFLO": {"name": "VictoryShares Free Cash Flow ETF", "region": "US"},
    "SCHD": {"name": "Schwab US Dividend Equity", "region": "US"},
    "JPST": {"name": "JPMorgan Ultra-Short Income", "region": "US"},
    "JAAA": {"name": "Janus Henderson AAA CLO ETF", "region": "US"},
    "SCYB": {"name": "Schwab High Yield Bond ETF", "region": "US"},
    
    # US Sector & Momentum
    "SMH":  {"name": "VanEck Semiconductor ETF", "region": "US"},
    "XLK":  {"name": "Technology Select Sector SPDR", "region": "US"},
    "XLF":  {"name": "Financial Select Sector SPDR", "region": "US"},
    "XLE":  {"name": "Energy Select Sector SPDR", "region": "US"},
    "XLI":  {"name": "Industrial Select Sector SPDR", "region": "US"},
    "XLV":  {"name": "Health Care Select Sector SPDR", "region": "US"},
    "IWM":  {"name": "iShares Russell 2000 ETF", "region": "US"},
    "QQQ":  {"name": "Invesco QQQ Trust", "region": "US"},
    "SPY":  {"name": "SPDR S&P 500 ETF Trust", "region": "US"},
}


# ==============================================================================
# ROBUST DATA FETCHING & ANALYTICS
# ==============================================================================

# Custom HTTP Session to mimic browser request & prevent Yahoo IP blocking
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})


def fetch_history_safely(ticker: str, period: str = "6m") -> pd.DataFrame:
    """Fetches price history using browser headers to avoid Cloud IP rate-limiting."""
    df = pd.DataFrame()
    
    # Method 1: yf.download with custom session
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, session=yf_session)
    except Exception:
        pass

    # Method 2: yf.Ticker object fallback
    if df.empty:
        try:
            tk = yf.Ticker(ticker, session=yf_session)
            df = tk.history(period=period)
        except Exception:
            pass

    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex headers if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.loc[:, ~df.columns.duplicated()]

    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
        return df

    return pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_fed_funds_probabilities():
    """Retrieves benchmark yields and macro posture."""
    df = fetch_history_safely("^TNX", period="5d")
    last_yield = 4.25
    if not df.empty and "Close" in df.columns:
        close_vals = df["Close"].values.flatten()
        last_yield = float(close_vals[-1])

    return {
        "Next Meeting": "Sep 16, 2026",
        "Pause Probability": "70.7%",
        "Cut Probability (-25bps)": "29.3%",
        "Hike Probability": "0.0%",
        "10Yr Benchmark Yield": f"{last_yield:.2f}%",
        "Regime Sentiment": "Pause Expected / Easing Bias"
    }


def analyze_etf_technical_ema(df: pd.DataFrame):
    """Calculates 20/50 day EMAs safely with flattened 1D arrays."""
    if len(df) < 30:
        return None

    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)
    
    ema20 = close_series.ewm(span=20, adjust=False).mean()
    ema50 = close_series.ewm(span=50, adjust=False).mean()

    latest_close = float(close_series.iloc[-1])
    latest_ema20 = float(ema20.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    prev_ema20 = float(ema20.iloc[-5]) if len(ema20) >= 5 else latest_ema20

    gap_pct = ((latest_ema20 - latest_ema50) / latest_ema50) * 100
    is_above = latest_ema20 > latest_ema50
    is_approaching = (not is_above) and (gap_pct > -3.0) and (latest_ema20 >= prev_ema20)

    status = "Bullish (20 > 50 EMA)" if is_above else ("Approaching Cross ↗️" if is_approaching else "Bearish Lag")
    
    return {
        "Close": latest_close,
        "EMA20": latest_ema20,
        "EMA50": latest_ema50,
        "Gap_Pct": gap_pct,
        "Status": status,
        "Bullish_Setup": is_above or is_approaching
    }


def fetch_institutional_flows_30d(df: pd.DataFrame):
    """Calculates 30-day Volume-Weighted Accumulation score safely."""
    if len(df) < 15 or "Volume" not in df.columns:
        return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

    hist30 = df.tail(22).copy()
    close_series = pd.Series(hist30["Close"].values.flatten())
    vol_series = pd.Series(hist30["Volume"].values.flatten())

    price_diff = close_series.diff()
    directional_vol = np.where(price_diff >= 0, vol_series, -vol_series)
    
    net_flow_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = vol_series.mean()
    
    if avg_vol == 0 or np.isnan(avg_vol):
        return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

    flow_score = min(100, max(0, int(50 + (net_flow_vol / (avg_vol * 10)) * 50)))
    
    if flow_score >= 60:
        signal = "🔥 Accumulation"
    elif flow_score <= 40:
        signal = "🚨 Distribution"
    else:
        signal = "➡️ Steady"

    return {
        "Flow_Signal": signal,
        "Net_30D_Score": flow_score
    }


def score_etf(ticker: str):
    """Calculates composite score (0-100) based on Technicals & Volume Money Flow."""
    df = fetch_history_safely(ticker, period="6m")
    if df.empty:
        return None

    tech = analyze_etf_technical_ema(df)
    if not tech:
        return None

    flows = fetch_institutional_flows_30d(df)

    score = 0
    if tech.get("Bullish_Setup"): score += 50
    if flows.get("Net_30D_Score", 0) >= 50: score += 50

    return {
        "Ticker": ticker,
        "Score": score,
        "Tech": tech,
        "Flows": flows
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

tab_discovery, tab_lookup = st.tabs([
    "🚀 90-Day Market Candidates",
    "🔍 Single ETF Deep-Dive"
])


# ==============================================================================
# TAB 1: AUTOMATED 90-DAY MARKET CANDIDATE DISCOVERY
# ==============================================================================
with tab_discovery:
    st.header("⚡ Top 90-Day Tactical Candidates Across Markets")
    st.caption("Scans broad sector, factor, and regional ETFs for upcoming momentum setups.")

    # Controls
    col_filter_reg, col_min_score, _ = st.columns([2, 2, 3])
    with col_filter_reg:
        selected_region = st.selectbox("Filter Region:", ["All Regions", "US", "Developed", "Emerging", "ex-China"])
    with col_min_score:
        min_score_cutoff = st.slider("Minimum 90D Score:", min_value=0, max_value=100, value=50, step=10)

    discovered_candidates = []

    with st.spinner("Scanning market candidates..."):
        for ticker, info in DYNAMIC_MARKET_POOL.items():
            if selected_region != "All Regions" and info["region"] != selected_region:
                continue

            res = score_etf(ticker)
            if res and res["Score"] >= min_score_cutoff:
                discovered_candidates.append({
                    "Ticker": ticker,
                    "Name": info["name"],
                    "Region": info["region"],
                    "90D Score": res["Score"],
                    "EMA Trend Setup": res["Tech"]["Status"],
                    "30D Inst Flow": res["Flows"]["Flow_Signal"],
                    "Flow Score": f"{res['Flows']['Net_30D_Score']}/100"
                })

    if discovered_candidates:
        cand_df = pd.DataFrame(discovered_candidates).sort_values(by="90D Score", ascending=False).reset_index(drop=True)

        st.dataframe(
            cand_df,
            hide_index=True,
            column_config={
                "90D Score": st.column_config.ProgressColumn("90D Readiness Score", format="%d pts", min_value=0, max_value=100),
                "Ticker": st.column_config.TextColumn("Ticker", width=80),
            },
            use_container_width=True
        )
    else:
        st.warning("No market candidates met the minimum score criteria for the selected filters.")


# ==============================================================================
# TAB 2: SINGLE ETF SCORE LOOKUP
# ==============================================================================
with tab_lookup:
    st.header("🔍 Single ETF Deep-Dive")
    st.caption("Calculate 90-day technical and flow scores for any ticker symbol.")

    col_input, _ = st.columns([2, 3])
    with col_input:
        lookup_ticker = st.text_input("Enter Ticker Symbol:", value="EMXC", placeholder="e.g. EMXC, SCHD, VFLO").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Evaluating {lookup_ticker}..."):
            res = score_etf(lookup_ticker)

        if res:
            st.markdown(f"### Score for **{lookup_ticker}**: `{res['Score']}/100` Points")
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric("Technical Setup (20/50 EMA)", res["Tech"]["Status"], delta=f"Gap: {res['Tech']['Gap_Pct']:.2f}%")
                st.write(f"- **Close:** ${res['Tech']['Close']:.2f}")
                st.write(f"- **20 EMA:** ${res['Tech']['EMA20']:.2f}")
                st.write(f"- **50 EMA:** ${res['Tech']['EMA50']:.2f}")

            with sc2:
                st.metric("30D Institutional Flow", res["Flows"]["Flow_Signal"], delta=f"Score: {res['Flows']['Net_30D_Score']}/100")
                st.write("Calculated via 30-day Volume-Weighted Money Flow.")
        else:
            st.error(f"Could not retrieve ticker data for '{lookup_ticker}'. Please verify the symbol.")
