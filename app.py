"""
app.py
======
Weekly ETF Rule Configurator & Scoring Engine (10 Rules)
Optimized for Weekly Timeframe / Medium-to-Long Term Position Screening.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Weekly ETF Screener & Rule Engine",
    page_icon="⚙️",
    layout="wide"
)

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
# SAFE URL & SESSION STATE SYNCHRONIZATION
# ==============================================================================

def get_url_tickers() -> str:
    """Safely extracts 'tickers' parameter from URL query params."""
    try:
        raw_val = st.query_params.get("tickers", "")
        if isinstance(raw_val, list):
            return raw_val[0] if raw_val else ""
        return str(raw_val) if raw_val else ""
    except Exception:
        return ""

url_tickers_clean = get_url_tickers()

# Initialize session_state key EXACTLY ONCE before the widget renders
if "tickers_input_field" not in st.session_state:
    st.session_state["tickers_input_field"] = url_tickers_clean if url_tickers_clean else "SPY, SCHD, VFLO, QQQ"

# Sync session state back to URL query params
def sync_query_params():
    current_val = st.session_state.get("tickers_input_field", "").strip()
    if current_val:
        st.query_params["tickers"] = current_val
    elif "tickers" in st.query_params:
        del st.query_params["tickers"]

if "config_df_v2" not in st.session_state:
    st.session_state["config_df_v2"] = pd.DataFrame([
        {"Rule #": "Rule 1", "Rule Name": "Weekly Trend (10/30 EMA)", "My Weight": 15},
        {"Rule #": "Rule 2", "Rule Name": "12-Week Absolute Return", "My Weight": 10},
        {"Rule #": "Rule 3", "Rule Name": "Weekly OBV Trend", "My Weight": 10},
        {"Rule #": "Rule 4", "Rule Name": "12-Week Relative Strength", "My Weight": 15},
        {"Rule #": "Rule 5", "Rule Name": "Weekly MACD Alignment", "My Weight": 10},
        {"Rule #": "Rule 6", "Rule Name": "26-Week Max Drawdown", "My Weight": 12},
        {"Rule #": "Rule 7", "Rule Name": "52-Week High Proximity", "My Weight": 8},
        {"Rule #": "Rule 8", "Rule Name": "Weekly RSI Band Filter", "My Weight": 5},
        {"Rule #": "Rule 9", "Rule Name": "52-Week Sharpe Ratio", "My Weight": 10},
        {"Rule #": "Rule 10", "Rule Name": "12-Week Money Flow Index", "My Weight": 5},
    ])


# ==============================================================================
# DATA FETCHING & WEEKLY RESAMPLING
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weekly_etf_history(ticker: str) -> pd.DataFrame:
    ticker_clean = ticker.strip().upper()
    try:
        df = yf.download(
            ticker_clean,
            period="2y",
            progress=False,
            auto_adjust=True,
            threads=False,
            ignore_tz=True
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            
            if "Close" in df.columns and len(df) > 60:
                weekly_df = pd.DataFrame()
                weekly_df["Open"] = df["Open"].resample("W-FRI").first()
                weekly_df["High"] = df["High"].resample("W-FRI").max()
                weekly_df["Low"] = df["Low"].resample("W-FRI").min()
                weekly_df["Close"] = df["Close"].resample("W-FRI").last()
                weekly_df["Volume"] = df["Volume"].resample("W-FRI").sum() if "Volume" in df.columns else 0
                
                return weekly_df.dropna(subset=["Close"]).reset_index()
    except Exception:
        pass
    return pd.DataFrame()


# ==============================================================================
# TECHNICAL HELPER FUNCTIONS
# ==============================================================================

def calculate_weekly_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not rsi.empty else 50.0
    return float(val) if not pd.isna(val) else 50.0

def derive_action_signal(score: int) -> tuple[str, str, str]:
    if score >= 70:
        return "🟢 BUY", "success", "Strong multi-week structural momentum and institutional accumulation."
    elif score >= 45:
        return "🟡 HOLD", "warning", "Consolidation or neutral weekly trend. Maintain current positioning."
    else:
        return "🔴 SELL", "error", "Weekly technical indicators indicate structural trend decay or drawdown."


# ==============================================================================
# WEEKLY RULE ENGINE
# ==============================================================================

def evaluate_weekly_rules(ticker: str, df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    if df.empty or len(df) < 35:
        return None

    ticker_upper = ticker.strip().upper()
    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten()) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

    # 1. Weekly Trend
    ema_fast = close.ewm(span=params["ema_fast_w"], adjust=False).mean()
    ema_slow = close.ewm(span=params["ema_slow_w"], adjust=False).mean()
    latest_close = float(close.iloc[-1])
    fast_val = float(ema_fast.iloc[-1])
    slow_val = float(ema_slow.iloc[-1])
    rule_ma_passed = fast_val > slow_val
    comm_ma = f"**Data:** 10 Wk EMA (${fast_val:.2f}) vs 30 Wk EMA (${slow_val:.2f})\n\n**Expected Range:** 10 Wk EMA > 30 Wk EMA."

    # 2. Absolute Return
    lookback_weeks = min(params["perf_weeks"], len(close) - 1)
    past_close = float(close.iloc[-lookback_weeks])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]
    comm_perf = f"**Data:** {lookback_weeks}-Week Return: {period_return_pct:+.2f}%\n\n**Expected Range:** Target ≥ +{params['min_return_pct']}%."

    # 3. Weekly OBV Trend
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    obv = (volume * direction).cumsum()
    obv_sma20 = obv.rolling(window=20).mean()
    latest_obv = float(obv.iloc[-1]) if not obv.empty else 0.0
    latest_obv_sma = float(obv_sma20.iloc[-1]) if not obv_sma20.empty else 0.0
    rule_obv_passed = latest_obv > latest_obv_sma
    comm_obv = f"**Data:** OBV: {latest_obv:,.0f} vs 20 Wk OBV SMA: {latest_obv_sma:,.0f}\n\n**Expected Range:** Weekly OBV > 20 Wk OBV SMA."

    # 4. Relative Strength vs SPY
    alpha_pct = 0.0
    rule_rs_passed = False
    rs_display = "❌ Fail"
    
    if ticker_upper == "SPY":
        rs_display = "N/A"
        comm_rs = f"**Data:** N/A (Benchmark Baseline)"
    else:
        if not benchmark_df.empty and len(benchmark_df) >= lookback_weeks:
            bench_close = pd.Series(benchmark_df["Close"].values.flatten())
            bench_latest = float(bench_close.iloc[-1])
            bench_past = float(bench_close.iloc[-min(lookback_weeks, len(bench_close) - 1)])
            bench_return = ((bench_latest - bench_past) / bench_past) * 100
            alpha_pct = period_return_pct - bench_return
            rule_rs_passed = alpha_pct >= params["min_alpha_pct"]
            rs_display = "✅ Pass" if rule_rs_passed else "❌ Fail"
        comm_rs = f"**Data:** 12-Week Alpha vs SPY: {alpha_pct:+.2f}%\n\n**Expected Range:** ≥ +1.0% Alpha."

    # 5. Weekly MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    rule_macd_passed = latest_macd > latest_signal
    comm_macd = f"**Data:** MACD Line: {latest_macd:.2f} vs Signal Line: {latest_signal:.2f}\n\n**Expected Range:** MACD > Signal."

    # 6. Max Drawdown
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 26:
        tail_26 = close.tail(26)
        rolling_max = tail_26.cummax()
        drawdown = (tail_26 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]
    comm_dd = f"**Data:** 26-Week Max Drawdown: {max_dd_pct:.2f}%\n\n**Expected Range:** ≤ 12.0%."

    # 7. 52-Week High Proximity
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 52:
        high_52w = float(close.tail(52).max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]
    comm_52w = f"**Data:** Distance from 52W High: {dist_52w_high_pct:.2f}%\n\n**Expected Range:** ≤ 10.0%."

    # 8. Weekly RSI Band Filter
    rsi_val = calculate_weekly_rsi(close, period=14)
    rule_rsi_passed = (rsi_val >= params["min_rsi"]) and (rsi_val <= params["max_rsi"])
    comm_rsi = f"**Data:** 14-Week RSI: {rsi_val:.1f}\n\n**Expected Range:** 48 to 68."

    # 9. 52-Week Sharpe Ratio
    weekly_returns = close.pct_change().dropna()
    ann_return = weekly_returns.mean() * 52
    ann_std = weekly_returns.std() * np.sqrt(52)
    sharpe_ratio = (ann_return / ann_std) if ann_std > 0 else 0.0
    rule_sharpe_passed = sharpe_ratio >= params["min_sharpe"]
    comm_sharpe = f"**Data:** Annualized Sharpe Ratio: {sharpe_ratio:.2f}\n\n**Expected Range:** ≥ 0.50."

    # 10. 12-Week Money Flow Index
    hist_vol = volume.tail(12)
    hist_close = close.tail(12)
    p_diff = hist_close.diff()
    directional_vol = np.where(p_diff >= 0, hist_vol, -hist_vol)
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 6)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]
    comm_flow = f"**Data:** 12-Week Flow Index: {flow_score}/100\n\n**Expected Range:** 50-100."

    total_score = 0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_obv_passed: total_score += params["weight_obv"]
    if rule_rs_passed: total_score += params["weight_rs"]
    if rule_macd_passed: total_score += params["weight_macd"]
    if rule_dd_passed: total_score += params["weight_dd"]
    if rule_52w_passed: total_score += params["weight_52w"]
    if rule_rsi_passed: total_score += params["weight_rsi"]
    if rule_sharpe_passed: total_score += params["weight_sharpe"]
    if rule_flow_passed: total_score += params["weight_flow"]

    return {
        "Score": total_score,
        "Close": latest_close,
        "Pass_MA": rule_ma_passed, "Comm_MA": comm_ma,
        "Pass_Perf": rule_perf_passed, "Comm_Perf": comm_perf,
        "Pass_OBV": rule_obv_passed, "Comm_OBV": comm_obv,
        "Pass_RS": rule_rs_passed, "RS_Display": rs_display, "Comm_RS": comm_rs,
        "Pass_MACD": rule_macd_passed, "Comm_MACD": comm_macd,
        "Pass_DD": rule_dd_passed, "Comm_DD": comm_dd,
        "Pass_52W": rule_52w_passed, "Comm_52W": comm_52w,
        "Pass_RSI": rule_rsi_passed, "Comm_RSI": comm_rsi,
        "Pass_Sharpe": rule_sharpe_passed, "Comm_Sharpe": comm_sharpe,
        "Pass_Flow": rule_flow_passed, "Comm_Flow": comm_flow
    }


# ==============================================================================
# MODAL SCORECARD WINDOW (@st.dialog)
# ==============================================================================

@st.dialog("🔍 Weekly Scorecard Breakdown", width="large")
def show_scorecard_modal(ticker: str, benchmark_df: pd.DataFrame, params: dict):
    st.subheader(f"Weekly Scorecard: {ticker}")

    with st.spinner(f"Analyzing {ticker} on Weekly scale..."):
        df = fetch_weekly_etf_history(ticker)
        res = evaluate_weekly_rules(ticker, df, benchmark_df, params)

    if res is not None:
        action_label, action_type, action_desc = derive_action_signal(res["Score"])

        c_metric1, c_metric2 = st.columns([1, 1])
        with c_metric1:
            st.metric(label=f"Weekly Score for {ticker}", value=f"{res['Score']} / 100 Points")
        with c_metric2:
            if action_type == "success":
                st.success(f"### Signal: {action_label}\n{action_desc}")
            elif action_type == "warning":
                st.warning(f"### Signal: {action_label}\n{action_desc}")
            else:
                st.error(f"### Signal: {action_label}\n{action_desc}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("1. Weekly Trend", "✅ PASS" if res["Pass_MA"] else "❌ FAIL", delta=f"{params['weight_ma'] if res['Pass_MA'] else 0} / {params['weight_ma']} pts")
            st.info(res["Comm_MA"])
        with c2:
            st.metric("2. 12W Return", "✅ PASS" if res["Pass_Perf"] else "❌ FAIL", delta=f"{params['weight_perf'] if res['Pass_Perf'] else 0} / {params['weight_perf']} pts")
            st.info(res["Comm_Perf"])
        with c3:
            st.metric("3. Weekly OBV", "✅ PASS" if res["Pass_OBV"] else "❌ FAIL", delta=f"{params['weight_obv'] if res['Pass_OBV'] else 0} / {params['weight_obv']} pts")
            st.info(res["Comm_OBV"])

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("4. 12W Rel Strength", res["RS_Display"], delta=f"{params['weight_rs'] if res['Pass_RS'] else 0} / {params['weight_rs']} pts")
            st.info(res["Comm_RS"])
        with c5:
            st.metric("5. Weekly MACD", "✅ PASS" if res["Pass_MACD"] else "❌ FAIL", delta=f"{params['weight_macd'] if res['Pass_MACD'] else 0} / {params['weight_macd']} pts")
            st.info(res["Comm_MACD"])
        with c6:
            st.metric("6. 26W Drawdown", "✅ PASS" if res["Pass_DD"] else "❌ FAIL", delta=f"{params['weight_dd'] if res['Pass_DD'] else 0} / {params['weight_dd']} pts")
            st.info(res["Comm_DD"])

        st.markdown("---")
        c7, c8, c9 = st.columns(3)
        with c7:
            st.metric("7. 52W High Prox.", "✅ PASS" if res["Pass_52W"] else "❌ FAIL", delta=f"{params['weight_52w'] if res['Pass_52W'] else 0} / {params['weight_52w']} pts")
            st.info(res["Comm_52W"])
        with c8:
            st.metric("8. Weekly RSI", "✅ PASS" if res["Pass_RSI"] else "❌ FAIL", delta=f"{params['weight_rsi'] if res['Pass_RSI'] else 0} / {params['weight_rsi']} pts")
            st.info(res["Comm_RSI"])
        with c9:
            st.metric("9. 52W Sharpe", "✅ PASS" if res["Pass_Sharpe"] else "❌ FAIL", delta=f"{params['weight_sharpe'] if res['Pass_Sharpe'] else 0} / {params['weight_sharpe']} pts")
            st.info(res["Comm_Sharpe"])

        st.markdown("---")
        c10, _ = st.columns([1, 2])
        with c10:
            st.metric("10. Money Flow Index", "✅ PASS" if res["Pass_Flow"] else "❌ FAIL", delta=f"{params['weight_flow'] if res['Pass_Flow'] else 0} / {params['weight_flow']} pts")
            st.info(res["Comm_Flow"])
    else:
        st.error(f"Could not retrieve historical data for '{ticker}'.")


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Weekly Points Configurator")
    st.caption("Adjust weight allocations across weekly rules (Must sum to 100).")

    edited_df = st.data_editor(
        st.session_state["config_df_v2"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rule #": st.column_config.TextColumn("Rule #", disabled=True),
            "Rule Name": st.column_config.TextColumn("Rule Name", disabled=True),
            "My Weight": st.column_config.NumberColumn("My Weight", min_value=0, max_value=100, step=1, format="%d")
        },
        key="rule_weights_editor_sidebar"
    )

    st.session_state["config_df_v2"] = edited_df
    total_raw_points = int(edited_df["My Weight"].sum())
    is_points_valid = (total_raw_points == 100)

    st.markdown(f"### **Total Points:** `{total_raw_points}`")
    if is_points_valid:
        st.success("✅ **Weight total equals 100 pts.**")
    else:
        diff = 100 - total_raw_points
        action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
        st.error(f"⚠️ Total is **{total_raw_points} pts** ({action_str}).")

weights = edited_df["My Weight"].tolist()
RULE_PARAMS = {
    "ema_fast_w": 10, "ema_slow_w": 30, "weight_ma": int(weights[0]),
    "perf_weeks": 12, "min_return_pct": 2.0, "weight_perf": int(weights[1]),
    "weight_obv": int(weights[2]),
    "min_alpha_pct": 1.0, "weight_rs": int(weights[3]),
    "weight_macd": int(weights[4]),
    "max_drawdown_pct": 12.0, "weight_dd": int(weights[5]),
    "max_dist_52w_pct": 10.0, "weight_52w": int(weights[6]),
    "min_rsi": 48.0, "max_rsi": 68.0, "weight_rsi": int(weights[7]),
    "min_sharpe": 0.5, "weight_sharpe": int(weights[8]),
    "min_flow_score": 50.0, "weight_flow": int(weights[9])
}


# ==============================================================================
# MAIN INTERFACE
# ==============================================================================

st.title("🎯 Weekly ETF Screener & Analysis")

benchmark_df = fetch_weekly_etf_history("SPY")

if not is_points_valid:
    st.error(f"⚠️ Points allocation total is currently {total_raw_points} pts. Please balance weights to 100 in the ⚙️ Sidebar Configurator.")

# Text area bound directly to key
tickers_input = st.text_area(
    "Tickers to Score:",
    height=120,
    placeholder="Enter tickers separated by commas (e.g. SPY, SCHD, VFLO, QQQ)...",
    key="tickers_input_field",
    on_change=sync_query_params
)

btn_run_screen = st.button(
    "Run Ticker Screen 🚀", 
    type="primary", 
    disabled=not is_points_valid or not tickers_input.strip(),
    use_container_width=True
)

should_run = btn_run_screen or ("auto_ran_on_load" not in st.session_state and bool(tickers_input.strip()))

if should_run:
    st.session_state["auto_ran_on_load"] = True
    active_tickers = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(active_tickers):
        df = fetch_weekly_etf_history(ticker)
        eval_res = evaluate_weekly_rules(ticker, df, benchmark_df, RULE_PARAMS)
        
        if eval_res is not None:
            action_sig, _, _ = derive_action_signal(eval_res["Score"])
            results.append({
                "Ticker": ticker,
                "Action": action_sig,
                "Score": eval_res["Score"],
                "Price_Raw": eval_res['Close'],
                "Price": f"${eval_res['Close']:.2f}",
                "Trend": "✅ Pass" if eval_res["Pass_MA"] else "❌ Fail",
                "Return": "✅ Pass" if eval_res["Pass_Perf"] else "❌ Fail",
                "OBV": "✅ Pass" if eval_res["Pass_OBV"] else "❌ Fail",
                "Rel Strength": eval_res["RS_Display"],
                "MACD": "✅ Pass" if eval_res["Pass_MACD"] else "❌ Fail",
                "Drawdown": "✅ Pass" if eval_res["Pass_DD"] else "❌ Fail",
                "52W High": "✅ Pass" if eval_res["Pass_52W"] else "❌ Fail",
                "RSI Band": "✅ Pass" if eval_res["Pass_RSI"] else "❌ Fail",
                "Sharpe": "✅ Pass" if eval_res["Pass_Sharpe"] else "❌ Fail",
                "Flow": "✅ Pass" if eval_res["Pass_Flow"] else "❌ Fail",
            })
        
        progress_bar.progress((idx + 1) / len(active_tickers))

    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.session_state["last_screener_df"] = res_df
        st.session_state["last_screener_title"] = "Weekly Scoring Matrix Results"
    else:
        st.warning("Could not retrieve valid historical data for any provided tickers.")

if "last_screener_df" in st.session_state and not st.session_state["last_screener_df"].empty:
    st.subheader(st.session_state.get("last_screener_title", "Weekly Scoring Matrix Results"))
    st.caption("💡 Select any row to pop open its detailed Scorecard modal window.")

    screener_df = st.session_state["last_screener_df"]
    
    event = st.dataframe(
        screener_df.drop(columns=["Price_Raw"]),
        hide_index=True,
        column_order=[
            "Ticker", "Action", "Score", "Price", 
            "Trend", "Return", "OBV", "Rel Strength", "MACD", 
            "Drawdown", "52W High", "RSI Band", "Sharpe", "Flow"
        ],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Action": st.column_config.TextColumn("Signal", help="🟢 BUY (≥70), 🟡 HOLD (45-69), 🔴 SELL (<45)"),
            "Score": st.column_config.NumberColumn("Score", format="%d pts")
        },
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event and event.selection and event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_ticker = screener_df.iloc[selected_index]["Ticker"]
        show_scorecard_modal(selected_ticker, benchmark_df, RULE_PARAMS)
