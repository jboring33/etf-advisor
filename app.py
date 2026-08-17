"""
app.py
======
Modular ETF Rule Configurator & Scoring Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="Custom ETF Screener & Rule Engine",
    page_icon="⚙️",
    layout="wide"
)

# Custom Styling
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
# DATA FETCHING ENGINE (OPTIMIZED FOR CUSTOM WATCHLISTS)
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_etf_history(ticker: str) -> pd.DataFrame:
    """Fetches price history using yfinance with standard fallbacks."""
    ticker_clean = ticker.strip().upper()
    try:
        df = yf.download(
            ticker_clean,
            period="1y",
            progress=False,
            auto_adjust=True,
            threads=False,
            ignore_tz=True
        )
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            if "Close" in df.columns and len(df) > 30:
                return df.dropna(subset=["Close"]).reset_index()
    except Exception:
        pass
    return pd.DataFrame()


# ==============================================================================
# SCORING & RULE EVALUATION ENGINE
# ==============================================================================

def evaluate_rules(df: pd.DataFrame, params: dict):
    """Evaluates customizable momentum, trend, and volume rules on price data."""
    if df.empty or len(df) < params["ema_slow"]:
        return None

    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten()) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

    # Rule 1: Fast/Slow Moving Average Trend
    ema_fast = close.ewm(span=params["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=params["ema_slow"], adjust=False).mean()
    
    latest_close = float(close.iloc[-1])
    latest_fast = float(ema_fast.iloc[-1])
    latest_slow = float(ema_slow.iloc[-1])
    
    ma_gap_pct = ((latest_fast - latest_slow) / latest_slow) * 100
    rule_ma_passed = latest_fast > latest_slow

    # Rule 2: Minimum N-Day Performance Return
    lookback_days = min(params["perf_days"], len(close) - 1)
    past_close = float(close.iloc[-lookback_days])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]

    # Rule 3: Institutional Money Flow / Accumulation
    hist_vol = volume.tail(22)
    hist_close = close.tail(22)
    price_diff = hist_close.diff()
    directional_vol = np.where(price_diff >= 0, hist_vol, -hist_vol)
    
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 10)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]

    # Composite Score Calculation
    total_score = 0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_flow_passed: total_score += params["weight_flow"]

    return {
        "Score": total_score,
        "Close": latest_close,
        "Fast_EMA": latest_fast,
        "Slow_EMA": latest_slow,
        "MA_Gap": ma_gap_pct,
        "Period_Return": period_return_pct,
        "Flow_Score": flow_score,
        "Pass_MA": rule_ma_passed,
        "Pass_Perf": rule_perf_passed,
        "Pass_Flow": rule_flow_passed
    }


# ==============================================================================
# SIDEBAR: RULE & PARAMETER CONFIGURATION
# ==============================================================================

st.sidebar.header("⚙️ Rule Configuration")
st.sidebar.caption("Adjust thresholds to customize scoring logic.")

st.sidebar.subheader("1. Trend Rule (EMA)")
ema_fast_val = st.sidebar.number_input("Fast EMA Span (Days)", value=20, step=5)
ema_slow_val = st.sidebar.number_input("Slow EMA Span (Days)", value=50, step=5)
weight_ma_val = st.sidebar.slider("Trend Rule Weight (pts)", 0, 50, 40)

st.sidebar.subheader("2. Performance Rule")
perf_days_val = st.sidebar.number_input("Lookback Window (Days)", value=60, step=10)
min_return_val = st.sidebar.number_input("Min Required Return (%)", value=2.0, step=0.5)
weight_perf_val = st.sidebar.slider("Performance Rule Weight (pts)", 0, 50, 30)

st.sidebar.subheader("3. Money Flow Rule")
min_flow_val = st.sidebar.slider("Min Money Flow Score", 0, 100, 50)
weight_flow_val = st.sidebar.slider("Money Flow Rule Weight (pts)", 0, 50, 30)

# Consolidated Rules Parameters Dictionary
RULE_PARAMS = {
    "ema_fast": ema_fast_val,
    "ema_slow": ema_slow_val,
    "weight_ma": weight_ma_val,
    "perf_days": perf_days_val,
    "min_return_pct": min_return_val,
    "weight_perf": weight_perf_val,
    "min_flow_score": min_flow_val,
    "weight_flow": weight_flow_val
}

MAX_POSSIBLE_SCORE = weight_ma_val + weight_perf_val + weight_flow_val


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

st.title("🎯 Custom ETF Screener & Scoring Engine")

tab_screen, tab_single = st.tabs([
    "📊 Batch Universe Screener",
    "🔍 Single Symbol Scorecard"
])


# ==============================================================================
# TAB 1: BATCH UNIVERSE SCREENER
# ==============================================================================
with tab_screen:
    st.header("Custom Universe Screening")
    st.caption("Enter your list of ETFs to evaluate them against your active sidebar rules.")

    default_tickers = "VFLO, SCHD, SCYB, JPST, JAAA, VEA, DIVI, EMXC, SMH, XLK, QQQ, SPY"
    user_input = st.text_area(
        "Enter ETF Tickers (comma or space separated):",
        value=default_tickers,
        height=100
    )

    tickers_list = [t.strip().upper() for t in user_input.replace("\n", ",").split(",") if t.strip()]

    min_total_score = st.slider("Minimum Composite Score Filter:", 0, MAX_POSSIBLE_SCORE, int(MAX_POSSIBLE_SCORE * 0.5))

    if st.button("Run Universe Screen", type="primary"):
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(tickers_list):
            df = fetch_etf_history(ticker)
            eval_res = evaluate_rules(df, RULE_PARAMS)
            
            if eval_res and eval_res["Score"] >= min_total_score:
                results.append({
                    "Ticker": ticker,
                    "Total Score": eval_res["Score"],
                    "Price": f"${eval_res['Close']:.2f}",
                    "Trend Rule": "✅ Pass" if eval_res["Pass_MA"] else "❌ Fail",
                    "Return Rule": "✅ Pass" if eval_res["Pass_Perf"] else "❌ Fail",
                    "Flow Rule": "✅ Pass" if eval_res["Pass_Flow"] else "❌ Fail",
                    "Return (%)": f"{eval_res['Period_Return']:.2f}%",
                    "Flow Score": f"{eval_res['Flow_Score']}/100"
                })
            
            progress_bar.progress((idx + 1) / len(tickers_list))

        progress_bar.empty()

        if results:
            res_df = pd.DataFrame(results).sort_values(by="Total Score", ascending=False).reset_index(drop=True)
            st.dataframe(
                res_df,
                hide_index=True,
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Total Score",
                        format="%d pts",
                        min_value=0,
                        max_value=MAX_POSSIBLE_SCORE
                    )
                },
                use_container_width=True
            )
        else:
            st.warning("No ETFs passed the minimum composite score threshold.")


# ==============================================================================
# TAB 2: SINGLE SYMBOL SCORECARD
# ==============================================================================
with tab_single:
    st.header("Single ETF Rule Breakdown")
    st.caption("Inspect exactly how an individual symbol scores against each rule.")

    lookup_ticker = st.text_input("Enter Ticker Symbol:", value="EMXC").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Fetching and analyzing {lookup_ticker}..."):
            df = fetch_etf_history(lookup_ticker)
            res = evaluate_rules(df, RULE_PARAMS)

        if res:
            st.markdown(f"### Score for **{lookup_ticker}**: `{res['Score']} / {MAX_POSSIBLE_SCORE}` Points")

            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.subheader("1. Trend Rule")
                status_ma = "✅ PASS" if res["Pass_MA"] else "❌ FAIL"
                st.metric("Trend Status", status_ma, delta=f"Points: {RULE_PARAMS['weight_ma'] if res['Pass_MA'] else 0}")
                st.write(f"- **Fast EMA ({RULE_PARAMS['ema_fast']}D):** ${res['Fast_EMA']:.2f}")
                st.write(f"- **Slow EMA ({RULE_PARAMS['ema_slow']}D):** ${res['Slow_EMA']:.2f}")
                st.write(f"- **Spread:** {res['MA_Gap']:.2f}%")

            with c2:
                st.subheader("2. Performance Rule")
                status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
                st.metric("Performance Status", status_perf, delta=f"Points: {RULE_PARAMS['weight_perf'] if res['Pass_Perf'] else 0}")
                st.write(f"- **Lookback Window:** {RULE_PARAMS['perf_days']} Days")
                st.write(f"- **Actual Return:** {res['Period_Return']:.2f}%")
                st.write(f"- **Target Return:** ≥ {RULE_PARAMS['min_return_pct']:.2f}%")

            with c3:
                st.subheader("3. Money Flow Rule")
                status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
                st.metric("Flow Status", status_flow, delta=f"Points: {RULE_PARAMS['weight_flow'] if res['Pass_Flow'] else 0}")
                st.write(f"- **Flow Score:** {res['Flow_Score']} / 100")
                st.write(f"- **Target Score:** ≥ {RULE_PARAMS['min_flow_score']}")
        else:
            st.error(f"Could not retrieve historical data for '{lookup_ticker}'. Please verify the symbol.")
