"""
app.py
======
Modular ETF Rule Configurator & Scoring Engine
Features:
- Adjustable Technical (EMA), Absolute Return, Money Flow, Relative Strength (SPY), 
  Volume Expansion, Trailing Volatility/Drawdown, and 52-Week High Proximity Rules.
- Batch Screener for custom universes.
- Single Symbol Scorecard with blank initial input.
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
# DATA FETCHING ENGINE
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_etf_history(ticker: str) -> pd.DataFrame:
    """Fetches 1 year of daily price history using yfinance."""
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

def evaluate_rules(df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    """Evaluates 7 total technical, performance, volume, and volatility rules."""
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

    # Rule 2: Minimum N-Day Absolute Return
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

    # Rule 4: Relative Strength vs Benchmark (SPY)
    alpha_pct = 0.0
    rule_rs_passed = False
    if not benchmark_df.empty and len(benchmark_df) >= lookback_days:
        bench_close = pd.Series(benchmark_df["Close"].values.flatten())
        bench_latest = float(bench_close.iloc[-1])
        bench_past = float(bench_close.iloc[-min(lookback_days, len(bench_close) - 1)])
        bench_return = ((bench_latest - bench_past) / bench_past) * 100
        alpha_pct = period_return_pct - bench_return
        rule_rs_passed = alpha_pct >= params["min_alpha_pct"]

    # Rule 5: Relative Volume Expansion (5D Vol vs 50D Vol)
    vol_ratio = 1.0
    rule_vol_exp_passed = False
    if len(volume) >= 50:
        vol_5d = volume.tail(5).mean()
        vol_50d = volume.tail(50).mean()
        vol_ratio = (vol_5d / vol_50d) if vol_50d > 0 else 1.0
        rule_vol_exp_passed = vol_ratio >= params["min_vol_ratio"]

    # Rule 6: Max Trailing Drawdown Filter (Volatility Check)
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 60:
        tail_60 = close.tail(60)
        rolling_max = tail_60.cummax()
        drawdown = (tail_60 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]

    # Rule 7: Proximity to 52-Week High
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 120:
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]

    # Composite Score Calculation
    total_score = 0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_flow_passed: total_score += params["weight_flow"]
    if rule_rs_passed: total_score += params["weight_rs"]
    if rule_vol_exp_passed: total_score += params["weight_vol_exp"]
    if rule_dd_passed: total_score += params["weight_dd"]
    if rule_52w_passed: total_score += params["weight_52w"]

    return {
        "Score": total_score,
        "Close": latest_close,
        "Fast_EMA": latest_fast,
        "Slow_EMA": latest_slow,
        "MA_Gap": ma_gap_pct,
        "Period_Return": period_return_pct,
        "Flow_Score": flow_score,
        "Alpha_Pct": alpha_pct,
        "Vol_Ratio": vol_ratio,
        "Max_Drawdown": max_dd_pct,
        "Dist_52W_High": dist_52w_high_pct,
        "Pass_MA": rule_ma_passed,
        "Pass_Perf": rule_perf_passed,
        "Pass_Flow": rule_flow_passed,
        "Pass_RS": rule_rs_passed,
        "Pass_VolExp": rule_vol_exp_passed,
        "Pass_DD": rule_dd_passed,
        "Pass_52W": rule_52w_passed
    }


# ==============================================================================
# SIDEBAR: RULE & PARAMETER CONFIGURATION
# ==============================================================================

st.sidebar.header("⚙️ Rule Configuration")
st.sidebar.caption("Adjust thresholds to customize scoring logic.")

st.sidebar.subheader("1. Trend Rule (EMA)")
ema_fast_val = st.sidebar.number_input("Fast EMA Span (Days)", value=20, step=5)
ema_slow_val = st.sidebar.number_input("Slow EMA Span (Days)", value=50, step=5)
weight_ma_val = st.sidebar.slider("Trend Rule Weight (pts)", 0, 50, 20)

st.sidebar.subheader("2. Absolute Return")
perf_days_val = st.sidebar.number_input("Lookback Window (Days)", value=60, step=10)
min_return_val = st.sidebar.number_input("Min Required Return (%)", value=2.0, step=0.5)
weight_perf_val = st.sidebar.slider("Performance Weight (pts)", 0, 50, 15)

st.sidebar.subheader("3. Money Flow Rule")
min_flow_val = st.sidebar.slider("Min Money Flow Score", 0, 100, 50)
weight_flow_val = st.sidebar.slider("Money Flow Weight (pts)", 0, 50, 15)

st.sidebar.subheader("4. Relative Strength vs SPY")
min_alpha_val = st.sidebar.number_input("Min Excess Alpha vs SPY (%)", value=1.0, step=0.5)
weight_rs_val = st.sidebar.slider("Relative Strength Weight (pts)", 0, 50, 15)

st.sidebar.subheader("5. Volume Expansion")
min_vol_ratio_val = st.sidebar.number_input("Min Volume Ratio (5D/50D)", value=1.1, step=0.1)
weight_vol_exp_val = st.sidebar.slider("Volume Expansion Weight (pts)", 0, 50, 10)

st.sidebar.subheader("6. Trailing Max Drawdown Filter")
max_dd_val = st.sidebar.number_input("Max Allowed Drawdown (%)", value=10.0, step=1.0)
weight_dd_val = st.sidebar.slider("Drawdown Weight (pts)", 0, 50, 15)

st.sidebar.subheader("7. Proximity to 52-Week High")
max_dist_52w_val = st.sidebar.number_input("Max % Distance from High", value=8.0, step=1.0)
weight_52w_val = st.sidebar.slider("52-Wk High Weight (pts)", 0, 50, 10)

# Consolidated Rules Parameters Dictionary
RULE_PARAMS = {
    "ema_fast": ema_fast_val,
    "ema_slow": ema_slow_val,
    "weight_ma": weight_ma_val,
    "perf_days": perf_days_val,
    "min_return_pct": min_return_val,
    "weight_perf": weight_perf_val,
    "min_flow_score": min_flow_val,
    "weight_flow": weight_flow_val,
    "min_alpha_pct": min_alpha_val,
    "weight_rs": weight_rs_val,
    "min_vol_ratio": min_vol_ratio_val,
    "weight_vol_exp": weight_vol_exp_val,
    "max_drawdown_pct": max_dd_val,
    "weight_dd": weight_dd_val,
    "max_dist_52w_pct": max_dist_52w_val,
    "weight_52w": weight_52w_val
}

MAX_POSSIBLE_SCORE = sum([
    weight_ma_val, weight_perf_val, weight_flow_val, 
    weight_rs_val, weight_vol_exp_val, weight_dd_val, weight_52w_val
])


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

st.title("🎯 Custom ETF Screener & Scoring Engine")

# Pre-fetch Benchmark (SPY) Data
benchmark_df = fetch_etf_history("SPY")

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
            eval_res = evaluate_rules(df, benchmark_df, RULE_PARAMS)
            
            if eval_res and eval_res["Score"] >= min_total_score:
                results.append({
                    "Ticker": ticker,
                    "Total Score": eval_res["Score"],
                    "Price": f"${eval_res['Close']:.2f}",
                    "Trend": "✅ Pass" if eval_res["Pass_MA"] else "❌ Fail",
                    "Return": "✅ Pass" if eval_res["Pass_Perf"] else "❌ Fail",
                    "Flow": "✅ Pass" if eval_res["Pass_Flow"] else "❌ Fail",
                    "Rel Strength": "✅ Pass" if eval_res["Pass_RS"] else "❌ Fail",
                    "Vol Exp": "✅ Pass" if eval_res["Pass_VolExp"] else "❌ Fail",
                    "Drawdown": "✅ Pass" if eval_res["Pass_DD"] else "❌ Fail",
                    "52W High": "✅ Pass" if eval_res["Pass_52W"] else "❌ Fail",
                    "Alpha vs SPY": f"{eval_res['Alpha_Pct']:+.2f}%",
                    "Max Drawdown": f"{eval_res['Max_Drawdown']:.2f}%",
                    "Off 52W High": f"-{eval_res['Dist_52W_High']:.2f}%"
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
    st.caption("Inspect exactly how an individual symbol scores against each active rule.")

    lookup_ticker = st.text_input("Enter Ticker Symbol:", value="", placeholder="e.g. EMXC, VFLO, SCHD").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Fetching and analyzing {lookup_ticker}..."):
            df = fetch_etf_history(lookup_ticker)
            res = evaluate_rules(df, benchmark_df, RULE_PARAMS)

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
                st.subheader("2. Absolute Return")
                status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
                st.metric("Return Status", status_perf, delta=f"Points: {RULE_PARAMS['weight_perf'] if res['Pass_Perf'] else 0}")
                st.write(f"- **Lookback Window:** {RULE_PARAMS['perf_days']} Days")
                st.write(f"- **Actual Return:** {res['Period_Return']:.2f}%")
                st.write(f"- **Target Return:** ≥ {RULE_PARAMS['min_return_pct']:.2f}%")

            with c3:
                st.subheader("3. Money Flow")
                status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
                st.metric("Flow Status", status_flow, delta=f"Points: {RULE_PARAMS['weight_flow'] if res['Pass_Flow'] else 0}")
                st.write(f"- **Flow Score:** {res['Flow_Score']} / 100")
                st.write(f"- **Target Score:** ≥ {RULE_PARAMS['min_flow_score']}")

            st.markdown("---")
            c4, c5, c6 = st.columns(3)

            with c4:
                st.subheader("4. Relative Strength vs SPY")
                status_rs = "✅ PASS" if res["Pass_RS"] else "❌ FAIL"
                st.metric("Rel Strength Status", status_rs, delta=f"Points: {RULE_PARAMS['weight_rs'] if res['Pass_RS'] else 0}")
                st.write(f"- **Excess Alpha:** {res['Alpha_Pct']:+.2f}%")
                st.write(f"- **Target Alpha:** ≥ {RULE_PARAMS['min_alpha_pct']:.2f}%")

            with c5:
                st.subheader("5. Volume Expansion")
                status_vol = "✅ PASS" if res["Pass_VolExp"] else "❌ FAIL"
                st.metric("Volume Status", status_vol, delta=f"Points: {RULE_PARAMS['weight_vol_exp'] if res['Pass_VolExp'] else 0}")
                st.write(f"- **5D vs 50D Vol Ratio:** {res['Vol_Ratio']:.2f}x")
                st.write(f"- **Target Ratio:** ≥ {RULE_PARAMS['min_vol_ratio']:.2f}x")

            with c6:
                st.subheader("6. Trailing Drawdown")
                status_dd = "✅ PASS" if res["Pass_DD"] else "❌ FAIL"
                st.metric("Drawdown Status", status_dd, delta=f"Points: {RULE_PARAMS['weight_dd'] if res['Pass_DD'] else 0}")
                st.write(f"- **60D Max Drawdown:** {res['Max_Drawdown']:.2f}%")
                st.write(f"- **Allowed Limit:** ≤ {RULE_PARAMS['max_drawdown_pct']:.2f}%")

            st.markdown("---")
            c7, _ = st.columns([1, 2])
            with c7:
                st.subheader("7. Proximity to 52-Week High")
                status_52w = "✅ PASS" if res["Pass_52W"] else "❌ FAIL"
                st.metric("52W Proximity Status", status_52w, delta=f"Points: {RULE_PARAMS['weight_52w'] if res['Pass_52W'] else 0}")
                st.write(f"- **Distance Off High:** -{res['Dist_52W_High']:.2f}%")
                st.write(f"- **Max Allowed Distance:** ≤ {RULE_PARAMS['max_dist_52w_pct']:.2f}%")
        else:
            st.error(f"Could not retrieve historical data for '{lookup_ticker}'. Please verify the symbol.")
    else:
        st.info("👆 Enter a ticker symbol above to generate a rule evaluation breakdown.")
