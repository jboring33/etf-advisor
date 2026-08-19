"""
app.py
======
Modular ETF Rule Configurator & Scoring Engine (10 Rules)
Features:
- Explicit point allocation per rule in sidebar.
- Automatic weight normalization ensuring maximum total score is ALWAYS 100 points.
- Batch Universe Screener & Single Symbol Scorecard fully in sync across all 10 rules.
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
# SESSION STATE INITIALIZATION
# ==============================================================================

if "user_tickers" not in st.session_state:
    st.session_state["user_tickers"] = "VFLO, SCHD, SCYB, JPST, JAAA, VEA, DIVI, EMXC, SMH, XLK, QQQ, SPY"


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
# TECHNICAL HELPER FUNCTIONS
# ==============================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates 14-day ATR relative to Current Price."""
    if len(df) < period + 1 or "High" not in df.columns or "Low" not in df.columns:
        return 0.0
    high = pd.Series(df["High"].values.flatten())
    low = pd.Series(df["Low"].values.flatten())
    close = pd.Series(df["Close"].values.flatten())
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    latest_close = close.iloc[-1]
    return float((atr / latest_close) * 100) if latest_close > 0 else 0.0


# ==============================================================================
# SCORING & RULE EVALUATION ENGINE (100-POINT CAP)
# ==============================================================================

def evaluate_rules(df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    """Evaluates 10 technical rules and applies normalized point scaling."""
    if df.empty or len(df) < params["ema_slow"]:
        return None

    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten()) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

    # 1. Moving Average Trend
    ema_fast = close.ewm(span=params["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=params["ema_slow"], adjust=False).mean()
    latest_close = float(close.iloc[-1])
    latest_fast = float(ema_fast.iloc[-1])
    latest_slow = float(ema_slow.iloc[-1])
    ma_gap_pct = ((latest_fast - latest_slow) / latest_slow) * 100
    rule_ma_passed = latest_fast > latest_slow

    # 2. Absolute Return
    lookback_days = min(params["perf_days"], len(close) - 1)
    past_close = float(close.iloc[-lookback_days])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]

    # 3. Institutional Money Flow
    hist_vol = volume.tail(22)
    hist_close = close.tail(22)
    price_diff = hist_close.diff()
    directional_vol = np.where(price_diff >= 0, hist_vol, -hist_vol)
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 10)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]

    # 4. Relative Strength vs SPY Benchmark
    alpha_pct = 0.0
    rule_rs_passed = False
    if not benchmark_df.empty and len(benchmark_df) >= lookback_days:
        bench_close = pd.Series(benchmark_df["Close"].values.flatten())
        bench_latest = float(bench_close.iloc[-1])
        bench_past = float(bench_close.iloc[-min(lookback_days, len(bench_close) - 1)])
        bench_return = ((bench_latest - bench_past) / bench_past) * 100
        alpha_pct = period_return_pct - bench_return
        rule_rs_passed = alpha_pct >= params["min_alpha_pct"]

    # 5. Volume Expansion
    vol_ratio = 1.0
    rule_vol_exp_passed = False
    if len(volume) >= 50:
        vol_5d = volume.tail(5).mean()
        vol_50d = volume.tail(50).mean()
        vol_ratio = (vol_5d / vol_50d) if vol_50d > 0 else 1.0
        rule_vol_exp_passed = vol_ratio >= params["min_vol_ratio"]

    # 6. Max Trailing Drawdown
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 60:
        tail_60 = close.tail(60)
        rolling_max = tail_60.cummax()
        drawdown = (tail_60 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]

    # 7. Proximity to 52-Week High
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 120:
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]

    # 8. RSI Band Filter
    rsi_val = calculate_rsi(close, period=14)
    rule_rsi_passed = (rsi_val >= params["min_rsi"]) and (rsi_val <= params["max_rsi"])

    # 9. Sharpe Ratio
    daily_returns = close.pct_change().dropna()
    ann_return = daily_returns.mean() * 252
    ann_std = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (ann_return / ann_std) if ann_std > 0 else 0.0
    rule_sharpe_passed = sharpe_ratio >= params["min_sharpe"]

    # 10. ATR Squeeze
    atr_pct = calculate_atr_ratio(df, period=14)
    rule_atr_passed = atr_pct <= params["max_atr_pct"]

    # Score Calculation (100 Point Scaling)
    total_score = 0.0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_flow_passed: total_score += params["weight_flow"]
    if rule_rs_passed: total_score += params["weight_rs"]
    if rule_vol_exp_passed: total_score += params["weight_vol_exp"]
    if rule_dd_passed: total_score += params["weight_dd"]
    if rule_52w_passed: total_score += params["weight_52w"]
    if rule_rsi_passed: total_score += params["weight_rsi"]
    if rule_sharpe_passed: total_score += params["weight_sharpe"]
    if rule_atr_passed: total_score += params["weight_atr"]

    return {
        "Score": round(total_score, 1),
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
        "RSI": rsi_val,
        "Sharpe": sharpe_ratio,
        "ATR_Pct": atr_pct,
        "Pass_MA": rule_ma_passed,
        "Pass_Perf": rule_perf_passed,
        "Pass_Flow": rule_flow_passed,
        "Pass_RS": rule_rs_passed,
        "Pass_VolExp": rule_vol_exp_passed,
        "Pass_DD": rule_dd_passed,
        "Pass_52W": rule_52w_passed,
        "Pass_RSI": rule_rsi_passed,
        "Pass_Sharpe": rule_sharpe_passed,
        "Pass_ATR": rule_atr_passed
    }


# ==============================================================================
# SIDEBAR: EXPLICIT POINT ALLOCATION PER RULE (100 TOTAL PTS)
# ==============================================================================

st.sidebar.header("🎯 Points Allocation Per Rule")
st.sidebar.caption("Adjust sliders below to allocate points to each rule. Points automatically normalize to equal **100 Total Points**.")

# Explicit Point Sliders
st.sidebar.markdown("---")
st.sidebar.subheader("1. Trend Rule (EMA)")
ema_fast_val = st.sidebar.number_input("Fast EMA Span (Days)", value=20, step=5)
ema_slow_val = st.sidebar.number_input("Slow EMA Span (Days)", value=50, step=5)
raw_ma = st.sidebar.slider("Points Allocation: Rule 1", 0, 50, 15, key="p_ma")

st.sidebar.subheader("2. Absolute Return")
perf_days_val = st.sidebar.number_input("Lookback Window (Days)", value=60, step=10)
min_return_val = st.sidebar.number_input("Min Required Return (%)", value=2.0, step=0.5)
raw_perf = st.sidebar.slider("Points Allocation: Rule 2", 0, 50, 10, key="p_perf")

st.sidebar.subheader("3. Money Flow Rule")
min_flow_val = st.sidebar.slider("Min Money Flow Score", 0, 100, 50)
raw_flow = st.sidebar.slider("Points Allocation: Rule 3", 0, 50, 10, key="p_flow")

st.sidebar.subheader("4. Relative Strength vs SPY")
min_alpha_val = st.sidebar.number_input("Min Excess Alpha vs SPY (%)", value=1.0, step=0.5)
raw_rs = st.sidebar.slider("Points Allocation: Rule 4", 0, 50, 15, key="p_rs")

st.sidebar.subheader("5. Volume Expansion")
min_vol_ratio_val = st.sidebar.number_input("Min Volume Ratio (5D/50D)", value=1.1, step=0.1)
raw_vol_exp = st.sidebar.slider("Points Allocation: Rule 5", 0, 50, 10, key="p_vol")

st.sidebar.subheader("6. Trailing Max Drawdown Filter")
max_dd_val = st.sidebar.number_input("Max Allowed Drawdown (%)", value=10.0, step=1.0)
raw_dd = st.sidebar.slider("Points Allocation: Rule 6", 0, 50, 10, key="p_dd")

st.sidebar.subheader("7. Proximity to 52-Week High")
max_dist_52w_val = st.sidebar.number_input("Max % Distance from High", value=8.0, step=1.0)
raw_52w = st.sidebar.slider("Points Allocation: Rule 7", 0, 50, 10, key="p_52w")

st.sidebar.subheader("8. RSI Range Filter")
min_rsi_val = st.sidebar.number_input("Min RSI (Not Oversold)", value=45.0, step=5.0)
max_rsi_val = st.sidebar.number_input("Max RSI (Not Overbought)", value=70.0, step=5.0)
raw_rsi = st.sidebar.slider("Points Allocation: Rule 8", 0, 50, 10, key="p_rsi")

st.sidebar.subheader("9. Risk-Adjusted Sharpe Ratio")
min_sharpe_val = st.sidebar.number_input("Min Sharpe Ratio", value=0.5, step=0.1)
raw_sharpe = st.sidebar.slider("Points Allocation: Rule 9", 0, 50, 10, key="p_sharpe")

st.sidebar.subheader("10. Volatility Squeeze (ATR %)")
max_atr_val = st.sidebar.number_input("Max Allowed ATR % of Price", value=2.5, step=0.5)
raw_atr = st.sidebar.slider("Points Allocation: Rule 10", 0, 50, 10, key="p_atr")

# Sum and Normalize to exactly 100 Points
raw_sum = max(1, sum([raw_ma, raw_perf, raw_flow, raw_rs, raw_vol_exp, raw_dd, raw_52w, raw_rsi, raw_sharpe, raw_atr]))

w_ma = round((raw_ma / raw_sum) * 100, 1)
w_perf = round((raw_perf / raw_sum) * 100, 1)
w_flow = round((raw_flow / raw_sum) * 100, 1)
w_rs = round((raw_rs / raw_sum) * 100, 1)
w_vol_exp = round((raw_vol_exp / raw_sum) * 100, 1)
w_dd = round((raw_dd / raw_sum) * 100, 1)
w_52w = round((raw_52w / raw_sum) * 100, 1)
w_rsi = round((raw_rsi / raw_sum) * 100, 1)
w_sharpe = round((raw_sharpe / raw_sum) * 100, 1)
w_atr = round((raw_atr / raw_sum) * 100, 1)

RULE_PARAMS = {
    "ema_fast": ema_fast_val,
    "ema_slow": ema_slow_val,
    "weight_ma": w_ma,
    "perf_days": perf_days_val,
    "min_return_pct": min_return_val,
    "weight_perf": w_perf,
    "min_flow_score": min_flow_val,
    "weight_flow": w_flow,
    "min_alpha_pct": min_alpha_val,
    "weight_rs": w_rs,
    "min_vol_ratio": min_vol_ratio_val,
    "weight_vol_exp": w_vol_exp,
    "max_drawdown_pct": max_dd_val,
    "weight_dd": w_dd,
    "max_dist_52w_pct": max_dist_52w_val,
    "weight_52w": w_52w,
    "min_rsi": min_rsi_val,
    "max_rsi": max_rsi_val,
    "weight_rsi": w_rsi,
    "min_sharpe": min_sharpe_val,
    "weight_sharpe": w_sharpe,
    "max_atr_pct": max_atr_val,
    "weight_atr": w_atr
}


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

st.title("🎯 Custom ETF Screener & Scoring Engine")

benchmark_df = fetch_etf_history("SPY")

# Display Active Point Breakdown Expander
with st.expander("📊 Active Rule Points Allocation Table (Normalized to 100 Total Points)"):
    pts_df = pd.DataFrame([
        {"Rule #": "Rule 1", "Rule Name": "Moving Average Trend", "Raw Slider Weight": raw_ma, "Normalized Points": f"{w_ma} pts"},
        {"Rule #": "Rule 2", "Rule Name": "Absolute Return", "Raw Slider Weight": raw_perf, "Normalized Points": f"{w_perf} pts"},
        {"Rule #": "Rule 3", "Rule Name": "Institutional Money Flow", "Raw Slider Weight": raw_flow, "Normalized Points": f"{w_flow} pts"},
        {"Rule #": "Rule 4", "Rule Name": "Relative Strength vs SPY", "Raw Slider Weight": raw_rs, "Normalized Points": f"{w_rs} pts"},
        {"Rule #": "Rule 5", "Rule Name": "Volume Expansion", "Raw Slider Weight": raw_vol_exp, "Normalized Points": f"{w_vol_exp} pts"},
        {"Rule #": "Rule 6", "Rule Name": "Max Trailing Drawdown", "Raw Slider Weight": raw_dd, "Normalized Points": f"{w_dd} pts"},
        {"Rule #": "Rule 7", "Rule Name": "52-Week High Proximity", "Raw Slider Weight": raw_52w, "Normalized Points": f"{w_52w} pts"},
        {"Rule #": "Rule 8", "Rule Name": "RSI Band Filter", "Raw Slider Weight": raw_rsi, "Normalized Points": f"{w_rsi} pts"},
        {"Rule #": "Rule 9", "Rule Name": "Sharpe Ratio Filter", "Raw Slider Weight": raw_sharpe, "Normalized Points": f"{w_sharpe} pts"},
        {"Rule #": "Rule 10", "Rule Name": "ATR Volatility Squeeze", "Raw Slider Weight": raw_atr, "Normalized Points": f"{w_atr} pts"},
    ])
    st.dataframe(pts_df, hide_index=True, use_container_width=True)

tab_screen, tab_single = st.tabs([
    "📊 Batch Universe Screener",
    "🔍 Single Symbol Scorecard"
])


# ==============================================================================
# TAB 1: BATCH UNIVERSE SCREENER
# ==============================================================================
with tab_screen:
    st.header("Custom Universe Screening")
    st.caption("Screen tickers scored against all 10 rules on a 100-point scale.")

    user_input = st.text_area(
        "Enter ETF Tickers (comma or space separated):",
        value=st.session_state["user_tickers"],
        height=100,
        key="ticker_input_field"
    )

    st.session_state["user_tickers"] = user_input
    tickers_list = [t.strip().upper() for t in user_input.replace("\n", ",").split(",") if t.strip()]

    min_total_score = st.slider("Minimum Composite Score Filter (out of 100):", 0, 100, 50)

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
                    "RSI Band": "✅ Pass" if eval_res["Pass_RSI"] else "❌ Fail",
                    "Sharpe": "✅ Pass" if eval_res["Pass_Sharpe"] else "❌ Fail",
                    "ATR Squeeze": "✅ Pass" if eval_res["Pass_ATR"] else "❌ Fail",
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
                        format="%.1f pts",
                        min_value=0,
                        max_value=100
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
    st.caption("Detailed breakdown of assigned points per rule out of 100 points.")

    lookup_ticker = st.text_input("Enter Ticker Symbol:", value="", placeholder="e.g. EMXC, VFLO, SCHD").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Fetching and analyzing {lookup_ticker}..."):
            df = fetch_etf_history(lookup_ticker)
            res = evaluate_rules(df, benchmark_df, RULE_PARAMS)

        if res:
            st.markdown(f"### Composite Score for **{lookup_ticker}**: `{res['Score']} / 100` Points")

            c1, c2, c3 = st.columns(3)
            with c1:
                status_ma = "✅ PASS" if res["Pass_MA"] else "❌ FAIL"
                pts_ma = RULE_PARAMS['weight_ma'] if res['Pass_MA'] else 0.0
                st.metric("1. Trend Status", status_ma, delta=f"{pts_ma} / {RULE_PARAMS['weight_ma']} pts")
                st.write(f"- Fast EMA ({RULE_PARAMS['ema_fast']}D): ${res['Fast_EMA']:.2f}")
                st.write(f"- Slow EMA ({RULE_PARAMS['ema_slow']}D): ${res['Slow_EMA']:.2f}")

            with c2:
                status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
                pts_perf = RULE_PARAMS['weight_perf'] if res['Pass_Perf'] else 0.0
                st.metric("2. Return Status", status_perf, delta=f"{pts_perf} / {RULE_PARAMS['weight_perf']} pts")
                st.write(f"- Return: {res['Period_Return']:.2f}%")
                st.write(f"- Min Target: {RULE_PARAMS['min_return_pct']}%")

            with c3:
                status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
                pts_flow = RULE_PARAMS['weight_flow'] if res['Pass_Flow'] else 0.0
                st.metric("3. Flow Status", status_flow, delta=f"{pts_flow} / {RULE_PARAMS['weight_flow']} pts")
                st.write(f"- Flow Score: {res['Flow_Score']} / 100")
                st.write(f"- Target: ≥ {RULE_PARAMS['min_flow_score']}")

            st.markdown("---")
            c4, c5, c6 = st.columns(3)
            with c4:
                status_rs = "✅ PASS" if res["Pass_RS"] else "❌ FAIL"
                pts_rs = RULE_PARAMS['weight_rs'] if res['Pass_RS'] else 0.0
                st.metric("4. Rel Strength", status_rs, delta=f"{pts_rs} / {RULE_PARAMS['weight_rs']} pts")
                st.write(f"- Alpha: {res['Alpha_Pct']:+.2f}%")
                st.write(f"- Target Alpha: ≥ {RULE_PARAMS['min_alpha_pct']:.2f}%")

            with c5:
                status_vol = "✅ PASS" if res["Pass_VolExp"] else "❌ FAIL"
                pts_vol = RULE_PARAMS['weight_vol_exp'] if res['Pass_VolExp'] else 0.0
                st.metric("5. Volume Ratio", status_vol, delta=f"{pts_vol} / {RULE_PARAMS['weight_vol_exp']} pts")
                st.write(f"- 5D/50D Ratio: {res['Vol_Ratio']:.2f}x")
                st.write(f"- Min Target: ≥ {RULE_PARAMS['min_vol_ratio']:.2f}x")

            with c6:
                status_dd = "✅ PASS" if res["Pass_DD"] else "❌ FAIL"
                pts_dd = RULE_PARAMS['weight_dd'] if res['Pass_DD'] else 0.0
                st.metric("6. Max Drawdown", status_dd, delta=f"{pts_dd} / {RULE_PARAMS['weight_dd']} pts")
                st.write(f"- 60D Drawdown: {res['Max_Drawdown']:.2f}%")
                st.write(f"- Max Limit: ≤ {RULE_PARAMS['max_drawdown_pct']:.2f}%")

            st.markdown("---")
            c7, c8, c9 = st.columns(3)
            with c7:
                status_52w = "✅ PASS" if res["Pass_52W"] else "❌ FAIL"
                pts_52w = RULE_PARAMS['weight_52w'] if res['Pass_52W'] else 0.0
                st.metric("7. 52W Proximity", status_52w, delta=f"{pts_52w} / {RULE_PARAMS['weight_52w']} pts")
                st.write(f"- Off 52W High: -{res['Dist_52W_High']:.2f}%")
                st.write(f"- Limit: ≤ {RULE_PARAMS['max_dist_52w_pct']:.2f}%")

            with c8:
                status_rsi = "✅ PASS" if res["Pass_RSI"] else "❌ FAIL"
                pts_rsi = RULE_PARAMS['weight_rsi'] if res['Pass_RSI'] else 0.0
                st.metric("8. RSI Band", status_rsi, delta=f"{pts_rsi} / {RULE_PARAMS['weight_rsi']} pts")
                st.write(f"- Current RSI: {res['RSI']:.1f}")
                st.write(f"- Allowed Range: {RULE_PARAMS['min_rsi']} - {RULE_PARAMS['max_rsi']}")

            with c9:
                status_sharpe = "✅ PASS" if res["Pass_Sharpe"] else "❌ FAIL"
                pts_sharpe = RULE_PARAMS['weight_sharpe'] if res['Pass_Sharpe'] else 0.0
                st.metric("9. Sharpe Ratio", status_sharpe, delta=f"{pts_sharpe} / {RULE_PARAMS['weight_sharpe']} pts")
                st.write(f"- Sharpe: {res['Sharpe']:.2f}")
                st.write(f"- Min Target: ≥ {RULE_PARAMS['min_sharpe']:.2f}")

            st.markdown("---")
            c10, _ = st.columns([1, 2])
            with c10:
                status_atr = "✅ PASS" if res["Pass_ATR"] else "❌ FAIL"
                pts_atr = RULE_PARAMS['weight_atr'] if res['Pass_ATR'] else 0.0
                st.metric("10. ATR Volatility", status_atr, delta=f"{pts_atr} / {RULE_PARAMS['weight_atr']} pts")
                st.write(f"- ATR %: {res['ATR_Pct']:.2f}%")
                st.write(f"- Max Limit: ≤ {RULE_PARAMS['max_atr_pct']:.2f}%")
        else:
            st.error(f"Could not retrieve historical data for '{lookup_ticker}'.")
