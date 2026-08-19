"""
app.py
======
Modular ETF Rule Configurator & Scoring Engine (10 Rules)
Features:
- Streamlined Points Configurator matching the 3-column format:
  [Rule #, Rule Name, My Weight]
- Session state cache reset to fix KeyError on schema change.
- Real-time 100-point total tracking and validation.
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

# Initialize weight table with updated key to overwrite stale session state
if "config_df_v2" not in st.session_state:
    st.session_state["config_df_v2"] = pd.DataFrame([
        {"Rule #": "Rule 1", "Rule Name": "Moving Average Trend", "My Weight": 12},
        {"Rule #": "Rule 2", "Rule Name": "Absolute Return", "My Weight": 8},
        {"Rule #": "Rule 3", "Rule Name": "Institutional Money Flow", "My Weight": 11},
        {"Rule #": "Rule 4", "Rule Name": "Relative Strength vs SPY", "My Weight": 15},
        {"Rule #": "Rule 5", "Rule Name": "Volume Expansion", "My Weight": 6},
        {"Rule #": "Rule 6", "Rule Name": "Max Trailing Drawdown", "My Weight": 16},
        {"Rule #": "Rule 7", "Rule Name": "52-Week High Proximity", "My Weight": 7},
        {"Rule #": "Rule 8", "Rule Name": "RSI Band Filter", "My Weight": 5},
        {"Rule #": "Rule 9", "Rule Name": "Sharpe Ratio Filter", "My Weight": 15},
        {"Rule #": "Rule 10", "Rule Name": "ATR Volatility Squeeze", "My Weight": 5},
    ])


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
# SCORING & RULE EVALUATION ENGINE
# ==============================================================================

def evaluate_rules(df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    """Evaluates 10 technical rules using raw weight assignments."""
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

    # Direct Weight Sum Scoring
    total_score = 0
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
# MAIN APPLICATION INTERFACE
# ==============================================================================

st.title("🎯 Custom ETF Screener & Scoring Engine")

benchmark_df = fetch_etf_history("SPY")

# Navigation Tabs
tab_config, tab_screen, tab_single = st.tabs([
    "⚙️ Points Configurator",
    "📊 Batch Universe Screener",
    "🔍 Single Symbol Scorecard"
])


# ==============================================================================
# TAB 1: POINTS CONFIGURATOR
# ==============================================================================
with tab_config:
    st.subheader("Rule Weight Allocation")
    
    edited_df = st.data_editor(
        st.session_state["config_df_v2"],
        hide_index=True,
        use_container_width=False,
        width=500,
        column_config={
            "Rule #": st.column_config.TextColumn("Rule #", disabled=True),
            "Rule Name": st.column_config.TextColumn("Rule Name", disabled=True),
            "My Weight": st.column_config.NumberColumn("My Weight", min_value=0, max_value=100, step=1, format="%d")
        },
        key="rule_weights_editor_v2"
    )

    st.session_state["config_df_v2"] = edited_df

    total_raw_points = int(edited_df["My Weight"].sum())
    is_points_valid = (total_raw_points == 100)

    st.markdown(f"### **Total:** `{total_raw_points}`")

    if is_points_valid:
        st.success("✅ **Total weight equals 100 points.**")
    else:
        diff = 100 - total_raw_points
        action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
        st.error(f"⚠️ Current total is **{total_raw_points} pts**. Adjust weights to reach **100** ({action_str}).")


# Extract weights directly from the table rows
weights = edited_df["My Weight"].tolist()
RULE_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "weight_ma": int(weights[0]),
    "perf_days": 60,
    "min_return_pct": 2.0,
    "weight_perf": int(weights[1]),
    "min_flow_score": 50.0,
    "weight_flow": int(weights[2]),
    "min_alpha_pct": 1.0,
    "weight_rs": int(weights[3]),
    "min_vol_ratio": 1.1,
    "weight_vol_exp": int(weights[4]),
    "max_drawdown_pct": 10.0,
    "weight_dd": int(weights[5]),
    "max_dist_52w_pct": 8.0,
    "weight_52w": int(weights[6]),
    "min_rsi": 45.0,
    "max_rsi": 70.0,
    "weight_rsi": int(weights[7]),
    "min_sharpe": 0.5,
    "weight_sharpe": int(weights[8]),
    "max_atr_pct": 2.5,
    "weight_atr": int(weights[9])
}


# Sidebar Status Display
st.sidebar.title("📊 System Status")
if is_points_valid:
    st.sidebar.success(f"**Total Points: {total_raw_points} / 100**\n\nRule set is balanced and valid.")
else:
    diff = 100 - total_raw_points
    action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
    st.sidebar.error(f"**Total Points: {total_raw_points} / 100**\n\nAction required: **{action_str}** in the Configurator table.")


# ==============================================================================
# TAB 2: BATCH UNIVERSE SCREENER
# ==============================================================================
with tab_screen:
    st.header("Custom Universe Screening")

    if not is_points_valid:
        st.error(
            f"⚠️ **Point allocation total is currently {total_raw_points} pts.** "
            f"Switch to the **⚙️ Points Configurator** tab to balance points to **100 total points**."
        )

    user_input = st.text_area(
        "Enter ETF Tickers (comma or space separated):",
        value=st.session_state["user_tickers"],
        height=100,
        key="ticker_input_field"
    )

    st.session_state["user_tickers"] = user_input
    tickers_list = [t.strip().upper() for t in user_input.replace("\n", ",").split(",") if t.strip()]

    min_total_score = st.slider("Minimum Composite Score Filter (out of 100):", 0, 100, 50)

    if st.button("Run Universe Screen", type="primary", disabled=not is_points_valid):
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
                        format="%d pts",
                        min_value=0,
                        max_value=100
                    )
                },
                use_container_width=True
            )
        else:
            st.warning("No ETFs passed the minimum composite score threshold.")


# ==============================================================================
# TAB 3: SINGLE SYMBOL SCORECARD
# ==============================================================================
with tab_single:
    st.header("Single ETF Rule Breakdown")

    lookup_ticker = st.text_input("Enter Ticker Symbol:", value="", placeholder="e.g. EMXC, VFLO, SCHD").strip().upper()

    if lookup_ticker:
        if not is_points_valid:
            st.warning("⚠️ Points allocation total must equal exactly 100 points. Please adjust rules in the Configurator table.")
        else:
            with st.spinner(f"Fetching and analyzing {lookup_ticker}..."):
                df = fetch_etf_history(lookup_ticker)
                res = evaluate_rules(df, benchmark_df, RULE_PARAMS)

            if res:
                st.markdown(f"### Composite Score for **{lookup_ticker}**: `{res['Score']} / 100` Points")

                c1, c2, c3 = st.columns(3)
                with c1:
                    status_ma = "✅ PASS" if res["Pass_MA"] else "❌ FAIL"
                    pts_ma = RULE_PARAMS['weight_ma'] if res['Pass_MA'] else 0
                    st.metric("1. Trend Status", status_ma, delta=f"{pts_ma} / {RULE_PARAMS['weight_ma']} pts")

                with c2:
                    status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
                    pts_perf = RULE_PARAMS['weight_perf'] if res['Pass_Perf'] else 0
                    st.metric("2. Return Status", status_perf, delta=f"{pts_perf} / {RULE_PARAMS['weight_perf']} pts")

                with c3:
                    status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
                    pts_flow = RULE_PARAMS['weight_flow'] if res['Pass_Flow'] else 0
                    st.metric("3. Flow Status", status_flow, delta=f"{pts_flow} / {RULE_PARAMS['weight_flow']} pts")

                st.markdown("---")
                c4, c5, c6 = st.columns(3)
                with c4:
                    status_rs = "✅ PASS" if res["Pass_RS"] else "❌ FAIL"
                    pts_rs = RULE_PARAMS['weight_rs'] if res['Pass_RS'] else 0
                    st.metric("4. Rel Strength", status_rs, delta=f"{pts_rs} / {RULE_PARAMS['weight_rs']} pts")

                with c5:
                    status_vol = "✅ PASS" if res["Pass_VolExp"] else "❌ FAIL"
                    pts_vol = RULE_PARAMS['weight_vol_exp'] if res['Pass_VolExp'] else 0
                    st.metric("5. Volume Ratio", status_vol, delta=f"{pts_vol} / {RULE_PARAMS['weight_vol_exp']} pts")

                with c6:
                    status_dd = "✅ PASS" if res["Pass_DD"] else "❌ FAIL"
                    pts_dd = RULE_PARAMS['weight_dd'] if res['Pass_DD'] else 0
                    st.metric("6. Max Drawdown", status_dd, delta=f"{pts_dd} / {RULE_PARAMS['weight_dd']} pts")

                st.markdown("---")
                c7, c8, c9 = st.columns(3)
                with c7:
                    status_52w = "✅ PASS" if res["Pass_52W"] else "❌ FAIL"
                    pts_52w = RULE_PARAMS['weight_52w'] if res['Pass_52W'] else 0
                    st.metric("7. 52W Proximity", status_52w, delta=f"{pts_52w} / {RULE_PARAMS['weight_52w']} pts")

                with c8:
                    status_rsi = "✅ PASS" if res["Pass_RSI"] else "❌ FAIL"
                    pts_rsi = RULE_PARAMS['weight_rsi'] if res['Pass_RSI'] else 0
                    st.metric("8. RSI Band", status_rsi, delta=f"{pts_rsi} / {RULE_PARAMS['weight_rsi']} pts")

                with c9:
                    status_sharpe = "✅ PASS" if res["Pass_Sharpe"] else "❌ FAIL"
                    pts_sharpe = RULE_PARAMS['weight_sharpe'] if res['Pass_Sharpe'] else 0
                    st.metric("9. Sharpe Ratio", status_sharpe, delta=f"{pts_sharpe} / {RULE_PARAMS['weight_sharpe']} pts")

                st.markdown("---")
                c10, _ = st.columns([1, 2])
                with c10:
                    status_atr = "✅ PASS" if res["Pass_ATR"] else "❌ FAIL"
                    pts_atr = RULE_PARAMS['weight_atr'] if res['Pass_ATR'] else 0
                    st.metric("10. ATR Volatility", status_atr, delta=f"{pts_atr} / {RULE_PARAMS['weight_atr']} pts")
            else:
                st.error(f"Could not retrieve historical data for '{lookup_ticker}'.")
